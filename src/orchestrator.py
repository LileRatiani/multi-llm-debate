import json
import time
from typing import Dict, List
from pydantic import BaseModel
from src.config import Config
from src.llm_client import generate_json_response

# ==========================================
# PYDANTIC SCHEMAS FOR JSON ENFORCEMENT
# ==========================================

class RolePreferenceSchema(BaseModel):
    role_preferences: List[str]
    confidence_by_role: Dict[str, float]
    reasoning: str

class InitialSolutionSchema(BaseModel):
    thought_process: str
    final_answer: str

class PeerReviewSchema(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    critical_errors: List[str]
    suggested_changes: List[str]
    overall_assessment: str

class ChangeMadeSchema(BaseModel):
    critique: str
    response: str
    accepted: bool

class RefinedSolutionSchema(BaseModel):
    changes_made: List[ChangeMadeSchema]
    refined_solution: str
    refined_answer: str
    confidence: float

class FinalJudgmentSchema(BaseModel):
    winner: str
    confidence: float
    reasoning: str

# ==========================================
# ORCHESTRATOR LOGIC
# ==========================================

class DebateOrchestrator:
    def __init__(self):
        self.candidate_agents = {
            "LLM_1": Config.SOLVER_1_MODEL,
            "LLM_2": Config.SOLVER_2_MODEL,
            "LLM_3": Config.SOLVER_3_MODEL,
            "LLM_4": Config.JUDGE_MODEL,
        }
        self.solvers = {
            "Solver_1": Config.SOLVER_1_MODEL,
            "Solver_2": Config.SOLVER_2_MODEL,
            "Solver_3": Config.SOLVER_3_MODEL
        }
        self.judge = Config.JUDGE_MODEL
        
        self.solver_temperatures = {
            "Solver_1": 0.8,
            "Solver_2": 0.5,
            "Solver_3": 0.2
        }

    def run_stage_0(self, problem_text: str) -> dict:
        """Stage 0: Each LLM self-assesses which debate role fits it best for this question."""
        print(f"\n--- [STAGE 0: ROLE SELF-ASSESSMENT] ---")

        system_prompt = (
            "You are one of four expert LLMs participating in a collaborative problem-solving debate. "
            "Before solving, each model must self-assess which role suits it best for the given problem.\n\n"
            "Available roles:\n"
            "- Solver_1, Solver_2, Solver_3: independently solve the problem with step-by-step reasoning\n"
            "- Judge: critically evaluate solver outputs and synthesize the best final answer\n\n"
            "You may also use the generic key 'Solver' in confidence_by_role if you are equally suited to any solver slot."
        )

        self_assessments = {}

        for agent_id, model_slug in self.candidate_agents.items():
            print(f"-> Requesting role self-assessment from {agent_id} ({model_slug})...")

            user_prompt = (
                f"Problem: {problem_text}\n\n"
                "Which role(s) do you believe you are best suited for on this problem?\n"
                "Return role_preferences as an ordered list (most preferred first), "
                "confidence_by_role with scores from 0.0 to 1.0 for each role you consider, "
                "and brief reasoning for your choices."
            )

            raw_response = generate_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=model_slug,
                response_schema=RolePreferenceSchema,
                temperature=0.3,
            )

            parsed_json = None
            for attempt in range(2):
                try:
                    candidate = json.loads(raw_response) if raw_response.strip() else {}
                    if not candidate.get("reasoning") and attempt == 0:
                        print(f"   [Warning] {agent_id} missing reasoning; retrying...")
                        raw_response = generate_json_response(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt + "\n\nYou must include a non-empty reasoning field.",
                            model_name=model_slug,
                            response_schema=RolePreferenceSchema,
                            temperature=0.3,
                        )
                        continue
                    if not candidate.get("reasoning"):
                        raise ValueError("Missing reasoning in role assessment")
                    parsed_json = candidate
                    break
                except Exception as parse_error:
                    if attempt == 0:
                        print(f"   [Warning] {agent_id} role assessment parse failed; retrying...")
                        raw_response = generate_json_response(
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                            model_name=model_slug,
                            response_schema=RolePreferenceSchema,
                            temperature=0.3,
                        )
                    else:
                        raise parse_error

            try:
                self_assessments[agent_id] = parsed_json
            except Exception as e:
                print(f"   [Error] {agent_id} produced unusable role assessment: {e}")
                self_assessments[agent_id] = None

        return self_assessments

    def _preference_rank(self, assessment: dict, role: str) -> int:
        """Lower rank means stronger preference. Unlisted roles sort last."""
        preferences = assessment.get("role_preferences", [])
        for index, preferred_role in enumerate(preferences):
            if preferred_role == role:
                return index
            if role.startswith("Solver") and preferred_role == "Solver":
                return index
        return len(preferences) + 100

    def _role_confidence(self, assessment: dict, role: str) -> float:
        confidence = assessment.get("confidence_by_role", {})
        if role in confidence:
            return float(confidence[role])
        if role.startswith("Solver") and "Solver" in confidence:
            return float(confidence["Solver"])
        return 0.0

    def _agent_sort_key(self, agent_id: str, assessment: dict, role: str) -> tuple:
        """Deterministic ranking: higher confidence wins, then preference order, then agent id."""
        return (
            self._role_confidence(assessment, role),
            -self._preference_rank(assessment, role),
            agent_id,
        )

    def run_stage_0_5(self, self_assessments: dict) -> dict:
        """Stage 0.5: Deterministically assign one Judge and three Solvers from self-assessments."""
        print(f"\n--- [STAGE 0.5: ALGORITHMIC ROLE ASSIGNMENT] ---")

        valid_agents = {
            agent_id: assessment
            for agent_id, assessment in self_assessments.items()
            if assessment is not None
        }

        if len(valid_agents) < 4:
            print("   [Warning] Not all agents returned valid assessments; using default role mapping.")
            assignment = {
                "Solver_1": "LLM_1",
                "Solver_2": "LLM_2",
                "Solver_3": "LLM_3",
                "Judge": "LLM_4",
            }
        else:
            judge_agent = max(
                valid_agents,
                key=lambda agent_id: self._agent_sort_key(agent_id, valid_agents[agent_id], "Judge"),
            )
            remaining_agents = [agent_id for agent_id in sorted(valid_agents) if agent_id != judge_agent]

            assignment = {"Judge": judge_agent}
            for solver_slot in ["Solver_1", "Solver_2", "Solver_3"]:
                best_agent = max(
                    remaining_agents,
                    key=lambda agent_id: self._agent_sort_key(agent_id, valid_agents[agent_id], solver_slot),
                )
                assignment[solver_slot] = best_agent
                remaining_agents.remove(best_agent)

        self.solvers = {
            solver_slot: self.candidate_agents[agent_id]
            for solver_slot, agent_id in assignment.items()
            if solver_slot.startswith("Solver")
        }
        self.judge = self.candidate_agents[assignment["Judge"]]

        print("-> Final role assignment:")
        for role, agent_id in assignment.items():
            print(f"   {role}: {agent_id} ({self.candidate_agents[agent_id]})")

        return assignment

    def run_stage_1(self, problem_text: str) -> dict:
        """Stage 1: Independent Answer Generation with basic error validation"""
        print(f"\n--- [STAGE 1: INITIAL ANSWER GENERATION] ---")
        
        system_prompt = (
            "You are an expert academic solver participating in a collaborative debate. "
            "Your goal is to solve the problem with absolute technical accuracy. "
            "Break down your solution step-by-step. "
            "Keep thought_process concise (under 400 words) while remaining complete."
        )
        
        initial_solutions = {}
        
        for agent_name, model_slug in self.solvers.items():
            temp = self.solver_temperatures.get(agent_name, 0.7)
            print(f"-> Requesting initial solution from {agent_name} ({model_slug})...")
            
            parsed_json = None
            for attempt in range(2):
                raw_response = generate_json_response(
                    system_prompt=system_prompt,
                    user_prompt=f"Problem: {problem_text}",
                    model_name=model_slug,
                    response_schema=InitialSolutionSchema,
                    temperature=temp
                )

                try:
                    candidate = json.loads(raw_response) if raw_response.strip() else {}
                    if not candidate.get("thought_process") or candidate["thought_process"] == "Failed to parse.":
                        raise ValueError("Empty or invalid content returned")
                    parsed_json = candidate
                    break
                except Exception as e:
                    if attempt == 0:
                        print(f"   [Warning] {agent_name} produced unusable output; retrying...")
                    else:
                        print(f"   [Error] {agent_name} produced unusable output: {e}")
                        initial_solutions[agent_name] = None

            if parsed_json is not None:
                initial_solutions[agent_name] = parsed_json
                
        return initial_solutions

    def run_stage_2(self, problem_text: str, initial_solutions: dict) -> dict:
        """Stage 2: Peer Review Round, skipping failed Stage 1 inputs"""
        print(f"\n--- [STAGE 2: PEER REVIEW ROUND] ---")
        
        peer_reviews = {name: {} for name in self.solvers.keys()}

        for reviewer_name, reviewer_model in self.solvers.items():
            for peer_name, peer_solution in initial_solutions.items():
                if reviewer_name == peer_name or peer_solution is None:
                    continue 
                
                print(f"-> {reviewer_name} is evaluating {peer_name}'s solution...")
                
                system_prompt = (
                    "You are an expert academic peer-reviewer. Evaluate the provided solution "
                    "for logical flaws, calculation errors, and edge-case oversights. "
                    "Be critical and specific."
                )
                
                user_prompt = (
                    f"Problem: {problem_text}\n\n"
                    f"Peer's Thought Process: {peer_solution.get('thought_process', '')}\n\n"
                    f"Peer's Final Answer: {peer_solution.get('final_answer', '')}\n\n"
                    "Evaluate this solution."
                )

                raw_response = generate_json_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_name=reviewer_model,
                    response_schema=PeerReviewSchema,
                    temperature=0.3 
                )

                try:
                    parsed_json = json.loads(raw_response)
                    peer_reviews[reviewer_name][f"review_of_{peer_name}"] = parsed_json
                except Exception:
                    print(f"   [Warning] {reviewer_name} failed to generate a review.")

        return peer_reviews

    def _collect_reviews_for_solver(self, peer_reviews: dict, solver_name: str) -> dict:
        """Gather the two peer reviews written about a given solver's solution."""
        reviews_received = {}
        for reviewer_name, reviews in peer_reviews.items():
            if reviewer_name == solver_name:
                continue
            review_key = f"review_of_{solver_name}"
            if review_key in reviews:
                reviews_received[reviewer_name] = reviews[review_key]
        return reviews_received

    def _normalize_refined_response(self, parsed_json: dict) -> dict:
        """Map common alternate field names to the expected Stage 3 schema."""
        if not parsed_json:
            return {}

        if "refined_solution" not in parsed_json and "refined_answer" not in parsed_json:
            for value in parsed_json.values():
                if isinstance(value, dict) and (
                    value.get("refined_solution") or value.get("refined_answer")
                    or value.get("solution") or value.get("answer")
                ):
                    parsed_json = value
                    break

        field_aliases = {
            "refined_solution": (
                "refined_solution", "refinedSolution", "solution",
                "refined_reasoning", "updated_solution", "thought_process",
            ),
            "refined_answer": (
                "refined_answer", "refinedAnswer", "final_answer",
                "answer", "updated_answer",
            ),
            "confidence": ("confidence", "confidence_score"),
            "changes_made": ("changes_made", "changesMade", "changes"),
        }

        normalized = {}
        for canonical, aliases in field_aliases.items():
            for key in aliases:
                value = parsed_json.get(key)
                if value is not None and value != "":
                    normalized[canonical] = value
                    break

        changes = normalized.get("changes_made", [])
        if isinstance(changes, list):
            normalized["changes_made"] = [
                {
                    "critique": item.get("critique") or item.get("criticism") or str(item),
                    "response": item.get("response") or item.get("reply") or "",
                    "accepted": bool(item.get("accepted", item.get("accept", False))),
                }
                for item in changes
                if isinstance(item, dict)
            ]
        else:
            normalized["changes_made"] = []

        if "confidence" not in normalized:
            normalized["confidence"] = 0.5

        return normalized

    def _request_refined_solution(
        self,
        solver_name: str,
        model_slug: str,
        system_prompt: str,
        user_prompt: str,
        strict: bool = False,
    ) -> dict:
        """Call the LLM for a refined solution, with optional stricter retry prompt."""
        prompt = user_prompt
        if strict:
            prompt += (
                "\n\nIMPORTANT: Return a single JSON object with exactly these keys: "
                "'changes_made' (list), 'refined_solution' (non-empty string with full reasoning), "
                "'refined_answer' (non-empty string), 'confidence' (number 0-1)."
            )

        raw_response = generate_json_response(
            system_prompt=system_prompt,
            user_prompt=prompt,
            model_name=model_slug,
            response_schema=RefinedSolutionSchema,
            temperature=self.solver_temperatures.get(solver_name, 0.5),
        )

        parsed_json = json.loads(raw_response) if raw_response.strip() else {}
        return self._normalize_refined_response(parsed_json), raw_response

    def run_stage_3(self, problem_text: str, initial_solutions: dict, peer_reviews: dict) -> dict:
        """Stage 3: Each Solver refines their solution based on peer feedback."""
        print(f"\n--- [STAGE 3: REFINEMENT BASED ON FEEDBACK] ---")

        system_prompt = (
            "You are an expert academic solver participating in a collaborative debate. "
            "You previously submitted an initial solution that has been peer-reviewed. "
            "Your task is to refine your solution by:\n"
            "- Addressing each critique from your peers explicitly\n"
            "- Accepting valid feedback and revising your reasoning where appropriate\n"
            "- Defending your original reasoning when a critique is incorrect\n"
            "- Producing an improved final solution with step-by-step reasoning"
        )

        refined_solutions = {}

        for solver_name, model_slug in self.solvers.items():
            original_solution = initial_solutions.get(solver_name)
            if original_solution is None:
                print(f"-> Skipping {solver_name}: no valid initial solution to refine.")
                refined_solutions[solver_name] = None
                continue

            reviews_received = self._collect_reviews_for_solver(peer_reviews, solver_name)
            if not reviews_received:
                print(f"-> Skipping {solver_name}: no peer reviews received.")
                refined_solutions[solver_name] = None
                continue

            print(f"-> {solver_name} is refining their solution ({model_slug})...")

            user_prompt = (
                f"Problem: {problem_text}\n\n"
                f"Your Original Thought Process: {original_solution.get('thought_process', '')}\n\n"
                f"Your Original Final Answer: {original_solution.get('final_answer', '')}\n\n"
                f"Peer Reviews of Your Solution:\n{json.dumps(reviews_received, indent=2)}\n\n"
                "For each meaningful critique in the peer reviews, add an entry to changes_made "
                "with the critique summarized, your response, and whether you accepted it (true/false). "
                "Then provide your refined_solution (full step-by-step reasoning) and refined_answer."
            )

            parsed_json = None
            last_raw_response = ""

            for attempt in range(2):
                parsed_json, last_raw_response = self._request_refined_solution(
                    solver_name=solver_name,
                    model_slug=model_slug,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    strict=(attempt > 0),
                )

                if parsed_json.get("refined_solution") and parsed_json.get("refined_answer"):
                    refined_solutions[solver_name] = parsed_json
                    break

                if parsed_json.get("refined_solution") and not parsed_json.get("refined_answer"):
                    parsed_json["refined_answer"] = original_solution.get("final_answer", "")
                    if parsed_json["refined_answer"]:
                        print(f"   [Warning] {solver_name} omitted refined_answer; using original answer as fallback.")
                        refined_solutions[solver_name] = parsed_json
                        break

                if attempt == 0:
                    print(f"   [Warning] {solver_name} returned incomplete refinement; retrying...")
                else:
                    preview = last_raw_response[:300].replace("\n", " ")
                    print(
                        f"   [Error] {solver_name} produced unusable refinement: "
                        f"missing refined_solution or refined_answer. Raw: {preview!r}"
                    )
                    refined_solutions[solver_name] = None

        return refined_solutions

    def _truncate_text(self, text, limit: int = 600) -> str:
        if text is None:
            return ""
        text = str(text).strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    def _compact_review(self, review: dict, item_limit: int = 2, text_limit: int = 120) -> dict:
        def trim_list(items):
            if not isinstance(items, list):
                return []
            return [
                self._truncate_text(item if isinstance(item, str) else str(item), text_limit)
                for item in items[:item_limit]
            ]

        return {
            "strengths": trim_list(review.get("strengths", [])),
            "weaknesses": trim_list(review.get("weaknesses", [])),
            "critical_errors": trim_list(review.get("critical_errors", [])),
            "overall_assessment": self._truncate_text(review.get("overall_assessment"), text_limit * 2),
        }

    def _build_judge_context(
        self,
        problem_text: str,
        initial_solutions: dict,
        peer_reviews: dict,
        refined_solutions: dict,
        reasoning_limit: int = 700,
        review_item_limit: int = 2,
    ) -> dict:
        """Build a compact summary for the judge to stay within API token limits."""
        solvers_summary = {}

        for solver_name in self.solvers:
            initial = initial_solutions.get(solver_name) or {}
            refined = refined_solutions.get(solver_name) or {}
            reviews_received = self._collect_reviews_for_solver(peer_reviews, solver_name)

            solvers_summary[solver_name] = {
                "initial_answer": initial.get("final_answer"),
                "initial_reasoning": self._truncate_text(initial.get("thought_process"), reasoning_limit),
                "refined_answer": refined.get("refined_answer") if refined else None,
                "refined_reasoning": self._truncate_text(
                    refined.get("refined_solution") if refined else None,
                    reasoning_limit,
                ),
                "refinement_confidence": refined.get("confidence") if refined else None,
                "peer_feedback": {
                    reviewer: self._compact_review(review, item_limit=review_item_limit)
                    for reviewer, review in reviews_received.items()
                },
            }

        return {
            "problem": problem_text,
            "solvers": solvers_summary,
        }

    def _normalize_winner_name(self, winner: str) -> str:
        """Map judge output to a canonical solver slot name."""
        if not winner:
            return ""

        normalized = str(winner).strip().replace("-", "_").replace(" ", "_")
        if normalized.lower().startswith("solver"):
            digits = "".join(char for char in normalized if char.isdigit())
            if digits:
                return f"Solver_{digits}"

        return normalized

    def _build_judgment_result(
        self,
        winner: str,
        confidence: float,
        reasoning: str,
        refined_solutions: dict,
        initial_solutions: dict,
    ) -> dict:
        final_answer = self._extract_final_answer(winner, refined_solutions, initial_solutions)
        if not final_answer:
            raise ValueError(f"Winner {winner} has no usable answer")

        return {
            "judgment": {
                "winner": winner,
                "confidence": confidence,
                "reasoning": reasoning,
            },
            "final_answer": final_answer,
            "winning_solution": refined_solutions.get(winner) or initial_solutions.get(winner),
        }

    def _fallback_judgment(self, refined_solutions: dict, initial_solutions: dict) -> dict:
        """Pick the solver with the highest refinement confidence when the judge API fails."""
        candidates = []
        for solver_name in self.solvers:
            answer = self._extract_final_answer(solver_name, refined_solutions, initial_solutions)
            if not answer:
                continue
            refined = refined_solutions.get(solver_name) or {}
            confidence = float(refined.get("confidence", 0.0))
            candidates.append((confidence, solver_name, answer))

        if not candidates:
            return {
                "judgment": None,
                "final_answer": None,
                "winning_solution": None,
            }

        confidence, winner, final_answer = max(candidates, key=lambda item: (item[0], item[1]))
        print(f"   [Fallback] Selected {winner} by highest refinement confidence ({confidence}).")

        return self._build_judgment_result(
            winner=winner,
            confidence=confidence,
            reasoning="Fallback selection after judge API failure: chose the solver with the highest refinement confidence.",
            refined_solutions=refined_solutions,
            initial_solutions=initial_solutions,
        )

    def _extract_final_answer(self, solver_name: str, refined_solutions: dict, initial_solutions: dict) -> str:
        """Get the best available answer for a solver (refined first, then initial)."""
        refined = refined_solutions.get(solver_name)
        if refined and refined.get("refined_answer"):
            return refined["refined_answer"]

        initial = initial_solutions.get(solver_name)
        if initial and initial.get("final_answer"):
            return initial["final_answer"]

        return ""

    def run_stage_4(
        self,
        problem_text: str,
        initial_solutions: dict,
        peer_reviews: dict,
        refined_solutions: dict,
    ) -> dict:
        """Stage 4: Judge evaluates all solutions and selects the winner."""
        print(f"\n--- [STAGE 4: FINAL JUDGMENT] ---")
        print(f"-> Judge ({self.judge}) is evaluating all solutions...")

        system_prompt = (
            "You are the final judge in a multi-LLM collaborative debate. "
            "You must impartially evaluate all solver submissions and select the single best answer. "
            "Consider initial reasoning quality, peer review feedback, and refinement quality. "
            "Pick the winner based on technical correctness and clarity."
        )

        compact_attempts = [
            (700, 2),
            (350, 1),
        ]

        for attempt, (reasoning_limit, review_item_limit) in enumerate(compact_attempts):
            judge_context = self._build_judge_context(
                problem_text=problem_text,
                initial_solutions=initial_solutions,
                peer_reviews=peer_reviews,
                refined_solutions=refined_solutions,
                reasoning_limit=reasoning_limit,
                review_item_limit=review_item_limit,
            )

            user_prompt = (
                f"Problem: {problem_text}\n\n"
                f"Debate Summary:\n{json.dumps(judge_context, indent=2)}\n\n"
                "Select the best solver. Set winner to exactly one of: Solver_1, Solver_2, Solver_3."
            )

            raw_response = generate_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_name=self.judge,
                response_schema=FinalJudgmentSchema,
                temperature=0.2,
            )

            try:
                parsed_json = json.loads(raw_response) if raw_response.strip() else {}
                winner = self._normalize_winner_name(parsed_json.get("winner", ""))
                reasoning = parsed_json.get("reasoning", "")
                confidence = float(parsed_json.get("confidence", 0.0))

                if winner not in self.solvers:
                    raise ValueError(f"Invalid winner: {parsed_json.get('winner')}")
                if not reasoning:
                    raise ValueError("Missing reasoning in judgment")

                result = self._build_judgment_result(
                    winner=winner,
                    confidence=confidence,
                    reasoning=reasoning,
                    refined_solutions=refined_solutions,
                    initial_solutions=initial_solutions,
                )

                print(f"-> Winner: {winner} (confidence: {confidence})")
                print(f"-> Final Answer: {result['final_answer']}")
                return result

            except Exception as e:
                if attempt < len(compact_attempts) - 1:
                    print(f"   [Warning] Judge attempt failed ({e}); retrying with smaller context...")
                else:
                    print(f"   [Warning] Judge failed after retries ({e}); using fallback selection.")

        return self._fallback_judgment(refined_solutions, initial_solutions)

    def run_full_debate(self, problem_text: str) -> dict:
        """Run the complete debate pipeline from Stage 0 through Stage 4."""
        stage_0_results = self.run_stage_0(problem_text)
        role_assignment = self.run_stage_0_5(stage_0_results)

        stage_1_results = None
        for attempt in range(2):
            stage_1_results = self.run_stage_1(problem_text)
            if any(v is not None for v in stage_1_results.values()):
                break
            print("   [!] Stage 1 failed entirely, retrying...")
            time.sleep(5)

        stage_2_results = self.run_stage_2(problem_text, stage_1_results)
        stage_3_results = self.run_stage_3(problem_text, stage_1_results, stage_2_results)
        stage_4_results = self.run_stage_4(
            problem_text, stage_1_results, stage_2_results, stage_3_results
        )

        return {
            "problem": problem_text,
            "role_assignment": role_assignment,
            "stage_0": stage_0_results,
            "stage_1": stage_1_results,
            "stage_2": stage_2_results,
            "stage_3": stage_3_results,
            "stage_4": stage_4_results,
            "final_answer": stage_4_results.get("final_answer"),
        }


if __name__ == "__main__":
    orchestrator = DebateOrchestrator()
    test_problem = "In how many ways can you tile a 3x8 rectangle with 2x1 dominoes?"

    debate_result = orchestrator.run_full_debate(test_problem)

    print("\n=== DEBATE COMPLETE ===")
    print(json.dumps(debate_result["stage_4"], indent=2))
    print(f"\nFinal Answer Returned to User: {debate_result['final_answer']}")