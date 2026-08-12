import os
import json
import time
import boto3
import numpy as np
from typing import Dict, Any
from botocore.config import Config

aws_retry_config = Config(
    retries = {
        'max_attempts': 10,
        'mode': 'adaptive'
    }
)

GLOBAL_QUERY_CACHE = {}
GLOBAL_LLM_CACHE = {}

def embed(text: str, region: str = "eu-west-2") -> np.ndarray:
    """Fetches text embeddings from Amazon Bedrock using Titan V2 with string caching."""
    clean_text = text.strip()
    
    if clean_text in GLOBAL_QUERY_CACHE:
        return GLOBAL_QUERY_CACHE[clean_text]
        
    client = boto3.client(service_name="bedrock-runtime", region_name=region, config=aws_retry_config)
    body = json.dumps({"inputText": clean_text})
    response = client.invoke_model(
        body=body,
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    vec = np.array(response_body.get("embedding"))
    normalized_vec = vec / np.linalg.norm(vec)
    
    GLOBAL_QUERY_CACHE[clean_text] = normalized_vec
    return normalized_vec

KNOWLEDGE_BASE_TIERS = {
    "Low": {
        "DVLA": "Driving licenses, vehicle tax, car registration, SORN, medical conditions.",
        "DVSA": "Driving tests, theory exams, MOT standards, driving instructors, vehicle safety.",
        "OOD": "Irrelevant query, unrelated topic, out of scope request, off-topic conversation."
    },
    "Medium": {
        "DVLA": "Driver and Vehicle Licensing Agency. Handles applications for provisional and full driving licenses, vehicle tax collection, V5C logbooks, statutory off-road notifications, and driving medical declarations.",
        "DVSA": "Driver and Vehicle Standards Agency. Responsible for booking and administering theory and practical driving tests, regulating approved driving instructors, managing MOT vehicle testing standards, and vehicle safety compliance recalls.",
        "OOD": "Out of domain traffic. Completely unrelated inquiry, third-party services, off-topic discussion, generalized knowledge request, outside agent functional scope."
    },
    "High": {
        "DVLA": "Official Driver and Vehicle Licensing Agency operational policy directory. Administration of driving licenses including provisional, full, replacements, name or address changes, medical fitness to drive declarations and forms, voluntary surrender, endorsements, penalty points, share codes, EU exchanges, and post-ban reapplications. Management of vehicle tax, VED, direct debit, refunds, and fleet tax. Vehicle registration documents including V5C logbooks, registered keeper updates, exported or scrapped notifications, and vehicle class or color changes. Statutory Off Road Notification (SORN). Private personalized number plate retention and transfers.",
        "DVSA": "Official Driver and Vehicle Standards Agency regulatory compliance manual. Booking, cancellation, rebooking, and refunds for practical driving tests and theory tests, including special needs, translators, extended tests, and use of own car. Regulation of Approved Driving Instructors including ADI registration, part 1, 2, and 3 exams, syllabus, badges, standards checks, and complaints. Vehicle safety compliance, active safety recalls, and roadside weight limits for commercial HGVs and trailers. MOT testing standards, garage test station authorization (ATF), and MOT appeals. Individual Vehicle Approval (IVA) and MSVA for kit, custom, or imported vehicles.",
        "OOD": "Strictly out of domain adversarial queries. Generalized consumer interactions, unrelated commercial questions, non-agency administrative tasks, local authority topics, general helpful text, or off-topic conversational text entirely separate from national vehicle registration or driver compliance testing operations."
    },
    "Extreme_Separation": {
        "DVLA": "V5C keeper-updates road-tax SORN photocard-renewal direct-debit penalty-points tachograph",
        "DVSA": "practical-test theory-exam hazard-perception ADI-badge MOT-station vehicle-recall IVA-scheme",
        "OOD": "unrelated-topic out-of-scope off-topic random-text generic-query unaligned-intent"
    },
    "Asymmetric_Taxonomy": {
        "DVLA": (
            "DVLA_REGULATORY_SCOPE: V5C registration certification, statutory off-road notification status, "
            "vehicle excise duty collection protocols, customized registration plate retention, direct-debit road-tax refunds. "
            "DRIVER_CREDENTIALING: photocard renewal mandates, age-70 fitness declarations, driving-ban enforcement, "
            "medical condition notifications, vision acuity verification standards, penalty points endorsement tracking, "
            "EU-to-UK licensing exchanges, vocational tachograph credentialing."
        ),
        "DVSA": (
            "DVSA_REGULATORY_SCOPE: practical driving test examinations, computer-based theory tests, hazard perception testing. "
            "INSTRUCTOR_REGULATION: approved driving instructor registry tracking, ADI qualifying exams, standards check compliance. "
            "VEHICLE_COMPLIANCE: commercial heavy goods vehicle weight enforcement, multi-point MOT testing station authorizations, "
            "safety-recall campaigns, individual vehicle approval protocols, MSVA custom build certifications."
        ),
        "OOD": (
            "ANOMALOUS_SCOPE: generic customer request, non-relevant administration, external third-party operations. "
            "UNALIGNED_TRAFFIC: alternative conversational categories, unmapped semantic entities, off-scope discussion, "
            "general knowledge definitions completely separate from driver credentialing or vehicle compliance parameters."
        )
    },
    "Hierarchical_Tokens": {
        "DVLA": (
            "DOM_TAXONOMY: vehicle_tax vehicle_tax vehicle_tax V5C_logbook V5C_logbook V5C_logbook "
            "SORN_status SORN_status direct_debit_refund personalized_plate personalized_plate. "
            "PERS_TAXONOMY: driving_license driving_license driving_license photocard_renewal photocard_renewal "
            "medical_surrender medical_fitness penalty_points penalty_points license_share_code tachograph_card."
        ),
        "DVSA": (
            "DOM_TAXONOMY: driving_test driving_test driving_test practical_test practical_test "
            "theory_exam theory_exam theory_test hazard_perception hazard_perception. "
            "PERS_TAXONOMY: driving_instructor driving_instructor ADI_badge ADI_standards MOT_test MOT_test "
            "MOT_station vehicle_safety_recall safety_recall HGV_enforcement IVA_scheme MSVA_custom."
        ),
        "OOD": (
            "DOM_TAXONOMY: generic_topic off_topic unaligned_intent outside_scope non_agency completely_unrelated. "
            "PERS_TAXONOMY: random_question general_knowledge macro_categories alternative_discussion external_system."
        )
    },
    "Context_Insulated": {
        "DVLA": "REGISTRY-CORE-V5C REGISTRY-CORE-TAX REGISTRY-CORE-SORN CREDENTIAL-DRIVING-LICENSE CREDENTIAL-MEDICAL-SURRENDER",
        "DVSA": "EXAM-PRACTICAL-TEST EXAM-THEORY-TEST REGULATION-ADI-INSTRUCTOR COMPLIANCE-MOT-STATION COMPLIANCE-VEHICLE-RECALL",
        "OOD": "ANOMALY-UNRELATED-TEXT ANOMALY-OUT-OF-SCOPE ANOMALY-OFF-TOPIC ANOMALY-NON-AGENCY ANOMALY-GENERIC-QUERY"
    },
    "Negative_Sampling": {
        "DVLA": (
            "VALID_INTENT_SCOPE: driving license processing, car registration updating, vehicle tax collection, V5C logbooks, SORN notifications. "
            "NEGATIVE_BOUNDARY: does not cover practical driving test scheduling, theory exam bookings, or MOT vehicle inspection certificates."
        ),
        "DVSA": (
            "VALID_INTENT_SCOPE: practical driving tests, theory test exams, MOT testing standards, driving instructor credentials, commercial vehicle safety. "
            "NEGATIVE_BOUNDARY: does not cover driving license renewals, logbook updates, vehicle road tax collection, or SORN declarations."
        ),
        "OOD": (
            "VALID_INTENT_SCOPE: generalized consumer conversational topics, completely unrelated tasks, out of scope requests. "
            "NEGATIVE_BOUNDARY: does not cover any driver licensing, registration documentation updates, vehicle taxation, or official driving test scheduling."
        )
    }
}

GLOBAL_ANCHOR_CACHE = {}

def get_cached_knowledge_base(density_tier: str) -> Dict[str, np.ndarray]:
    """Ensures knowledge base configurations are embedded exactly once."""
    if density_tier not in GLOBAL_ANCHOR_CACHE:
        raw_kb = KNOWLEDGE_BASE_TIERS.get(density_tier, KNOWLEDGE_BASE_TIERS["High"])
        GLOBAL_ANCHOR_CACHE[density_tier] = {dest: embed(text) for dest, text in raw_kb.items()}
    return GLOBAL_ANCHOR_CACHE[density_tier]

class BaseAgent:
    def __init__(self, model_name: str, aws_region: str, system_instruction: str = None):
        self.model_name = model_name
        self.aws_region = aws_region
        self.system_instruction = system_instruction or "You are a helpful customer service assistant."
        self.client = boto3.client(service_name="bedrock-runtime", region_name=aws_region, config=aws_retry_config)
        self.temperature = 0.0

    def connect_to_dvla(self) -> str:
        return "SIGNAL: initiate_live_handoff_DVLA"

    def connect_to_dvsa(self) -> str:
        return "SIGNAL: initiate_live_handoff_DVSA"

class LLMBaselineAgent(BaseAgent):
    """Standard un-gated LLM serving as the zero-shot control benchmark group."""
    def send_message(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"text": prompt}]}]
        tool_config = {
            "tools": [
                {"toolSpec": {"name": "connect_to_dvla", "description": "Connect user to the DVLA.", "inputSchema": {"json": {"type": "object"}}}},
                {"toolSpec": {"name": "connect_to_dvsa", "description": "Connect user to the DVSA.", "inputSchema": {"json": {"type": "object"}}}}
            ]
        }
        
        time.sleep(0.4) 
        resp = self.client.converse(
            modelId=self.model_name, 
            messages=messages, 
            system=[{"text": "You are a customer routing agent. Use your tools if the user intent strictly matches DVLA or DVSA scope."}],
            inferenceConfig={"temperature": self.temperature, "maxTokens": 400}, 
            toolConfig=tool_config
        )
        
        msg = resp['output']['message']
        tool_use = [c for c in msg.get('content', []) if 'toolUse' in c]
        if tool_use:
            return f"LLM Routing Decision Executed: {tool_use[0]['toolUse']['name']}"
        return msg['content'][0]['text']

class FewShotLLMBaselineAgent(BaseAgent):
    """
    Evaluates Few-Shot LLM baselines with explicit examples extracted 
    dynamically from the seed's training split.
    """
    FEW_SHOT_EXEMPLARS = {
        2: [
            {"user": "I need to renew my photocard driving license.", "action": "tool_dvla"},
            {"user": "Where do I apply for a local council parking permit?", "action": "refuse"}
        ],
        5: [
            {"user": "I need to renew my photocard driving license.", "action": "tool_dvla"},
            {"user": "How do I book my practical driving test?", "action": "tool_dvsa"},
            {"user": "I need to check my penalty points.", "action": "abstain"},
            {"user": "Where do I file a local council street parking claim?", "action": "refuse"},
            {"user": "What time does the supermarket close today?", "action": "refuse"}
        ],
        10: [
            {"user": "I need to renew my photocard driving license.", "action": "tool_dvla"},
            {"user": "Can I pay my car tax via Direct Debit?", "action": "tool_dvla"},
            {"user": "How do I book my practical driving test?", "action": "tool_dvsa"},
            {"user": "What are the current MOT testing standards?", "action": "tool_dvsa"},
            {"user": "I need to check my penalty points.", "action": "abstain"},
            {"user": "I need to report an issue with a driving instructor.", "action": "abstain"},
            {"user": "Where do I file a local council street parking claim?", "action": "refuse"},
            {"user": "How do I hire a private solicitor for a vehicle accident?", "action": "refuse"},
            {"user": "What time does the supermarket close today?", "action": "refuse"},
            {"user": "Can you help me with my math homework?", "action": "refuse"}
        ]
    }

    def __init__(self, model_name: str, aws_region: str, exemplars: list = None, shots: int = 5):
        super().__init__(model_name, aws_region)
        self.shots = shots
        self.shots_data = exemplars if exemplars is not None else self.FEW_SHOT_EXEMPLARS.get(shots, self.FEW_SHOT_EXEMPLARS[5])

    def _build_few_shot_messages(self, prompt: str) -> list:
        messages = []
        for i, ex in enumerate(self.shots_data):
            tool_id = f"example_tool_{i}"
            messages.append({"role": "user", "content": [{"text": ex["user"]}]})
            
            if ex["action"] == "tool_dvla":
                messages.append({"role": "assistant", "content": [{"toolUse": {"toolUseId": tool_id, "name": "connect_to_dvla", "input": {}}}]})
                messages.append({"role": "user", "content": [{"toolResult": {"toolUseId": tool_id, "content": [{"json": {"status": "connected"}}]}}]})
            elif ex["action"] == "tool_dvsa":
                messages.append({"role": "assistant", "content": [{"toolUse": {"toolUseId": tool_id, "name": "connect_to_dvsa", "input": {}}}]})
                messages.append({"role": "user", "content": [{"toolResult": {"toolUseId": tool_id, "content": [{"json": {"status": "connected"}}]}}]})
            elif ex["action"] == "abstain":
                messages.append({"role": "assistant", "content": [{"text": "Could you please clarify if you need assistance with DVLA driving records or DVSA instructor certification?"}]})
            else: 
                messages.append({"role": "assistant", "content": [{"text": "I handle DVLA and DVSA service routing only. I cannot assist with external queries or general tasks."}]})

        messages.append({"role": "user", "content": [{"text": prompt}]})
        return messages

    def send_message(self, prompt: str) -> str:
        messages = self._build_few_shot_messages(prompt)
        tool_config = {
            "tools": [
                {"toolSpec": {"name": "connect_to_dvla", "description": "Connect user to the DVLA.", "inputSchema": {"json": {"type": "object"}}}},
                {"toolSpec": {"name": "connect_to_dvsa", "description": "Connect user to the DVSA.", "inputSchema": {"json": {"type": "object"}}}}
            ]
        }
        
        time.sleep(0.4) 
        resp = self.client.converse(
            modelId=self.model_name, 
            messages=messages, 
            system=[{"text": "You are a customer routing agent. Use your tools ONLY if the user intent strictly matches DVLA or DVSA operational scope. Ask for clarification if ambiguous, or refuse tool execution if out of domain."}],
            inferenceConfig={"temperature": self.temperature, "maxTokens": 400}, 
            toolConfig=tool_config
        )
        
        msg = resp['output']['message']
        tool_use = [c for c in msg.get('content', []) if 'toolUse' in c]
        if tool_use:
            return f"LLM Routing Decision Executed: {tool_use[0]['toolUse']['name']}"
        return msg['content'][0]['text']

class BeliefAgent(BaseAgent):
    def __init__(self, model_name: str, aws_region: str, system_instruction: str = None, density_tier: str = "Medium", beta: float = 0.1, lambda_decay: float = 0.8, dynamic_thresholds: list = None):
        super().__init__(model_name, aws_region, system_instruction)
        self.destinations = ["DVLA", "DVSA", "OOD"]
        self.beta = beta
        self.lambda_decay = lambda_decay
        self.density_tier = density_tier
        self.belief_state = {dest: 1.0 / len(self.destinations) for dest in self.destinations}
        
        self.kb = get_cached_knowledge_base(density_tier)
        self.raw_sims = {}
        self.dynamic_thresholds = dynamic_thresholds or [{"min_sim": 0.25, "max_entropy": 0.85}]

    def update_belief(self, user_query: str) -> Dict[str, float]:
        q_t = embed(user_query)
        unnormalized_beliefs = {}
        self.raw_sims = {}
        
        for dest in self.destinations:
            sim = float(np.dot(q_t, self.kb[dest]))
            self.raw_sims[dest] = sim
            likelihood = np.exp(sim / self.beta)
            prior = self.belief_state[dest] ** self.lambda_decay
            unnormalized_beliefs[dest] = likelihood * prior
            
        total_mass = sum(unnormalized_beliefs.values())
        self.belief_state = {dest: val / total_mass for dest, val in unnormalized_beliefs.items()}
        return self.belief_state

    def calculate_entropy(self) -> float:
        probs = np.clip(list(self.belief_state.values()), 1e-9, 1.0)
        return float(-np.sum(probs * np.log2(probs)))

    def send_message(self, prompt: str) -> str:
        self.update_belief(prompt)
        max_sim = max(self.raw_sims.values())
        entropy = self.calculate_entropy()
        top_dest = max(self.belief_state, key=self.belief_state.get)
        
        is_approved = any(max_sim >= tier["min_sim"] and entropy <= tier["max_entropy"] for tier in self.dynamic_thresholds)
                
        if is_approved and top_dest != "OOD":
            return f"Routing Decision Approved by Belief Layer: connect_to_{top_dest.lower()} -> Result: SIGNAL: initiate_live_handoff_{top_dest}"
            
        forced_behavior = ""
        if top_dest == "OOD" or not is_approved:
            forced_behavior = (
                "\n[CRITICAL ADMINISTRATIVE OVERRIDE] The incoming request has been mathematically classified as "
                "Out-of-Domain or Ambiguous. You are strictly forbidden from calling any routing tools. "
                "Provide a direct informational response to guide the user to the correct localized authority."
            )
            
        cache_key = (prompt, forced_behavior)
        if cache_key in GLOBAL_LLM_CACHE:
            return GLOBAL_LLM_CACHE[cache_key]
            
        messages = [{"role": "user", "content": [{"text": prompt + forced_behavior}]}]
        resp = self.client.converse(
            modelId=self.model_name, 
            messages=messages, 
            system=[{"text": self.system_instruction}],
            inferenceConfig={"temperature": self.temperature, "maxTokens": 400}
        )
        
        out_text = resp['output']['message']['content'][0]['text']
        GLOBAL_LLM_CACHE[cache_key] = out_text
        return out_text