import os
import json
import time
import boto3
import random
import itertools
from tqdm import tqdm
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor, as_completed

OPENINGS = [
    "", "Hello. ", "Hi there. ", "Good morning. ", "Good afternoon. ",
    "I need some help. ", "Can you help me? ", "Quick question: ", "I have a question. ",
    "Please assist me. ", "Looking for some guidance. ", "Hi, ", "Greetings. ",
    "I am hoping you can help. ", "I've got an issue. ", "Need quick advice. ",
    "Excuse me, ", "To whom it may concern, ", "I require assistance. ", "Hey, "
]

# DVLA (Vehicle & Driver Registry)
DVLA_INTENTS = [
    "I need to pay the vehicle tax", "How do I declare a vehicle SORN", 
    "I lost my V5C logbook and need a replacement", "I need to renew my driving license", 
    "How do I put a private number plate on retention", "My driving license was stolen, how do I get a new one", 
    "I need to tax my motorcycle", "How much is the car tax", 
    "Can I pay my vehicle tax via Direct Debit", "I need to register an imported vehicle", 
    "I need to report a medical issue and surrender my license", "How do I update the registered keeper details", 
    "I need to change the address on my driving documents", "I need to exchange my EU driving license for a UK one", 
    "I need to check my penalty points", "How do I get a license check code for my employer",
    "I need to apply for my first provisional driving license", "How do I update my vehicle's color on the registry",
    "I am exporting my car and need to notify the registry", "How do I replace a damaged photocard license"
]
DVLA_CONTEXTS = [
    "for my new car.", "for a vehicle I just bought.", "because I am turning 70 soon.",
    "for my commercial delivery van.", "since I recently moved to a new house.",
    "for my disabled driver's adapted vehicle.", "for a used dealership car.",
    "regarding my heavy agricultural tractor.", "for my company fleet vehicle.",
    "involving a high-emissions hybrid vehicle.", "for an imported vintage motorcycle.",
    "because the original was destroyed in a fire.", "since I am going abroad for six months.",
    "for my weekend sports car.", "regarding a vehicle I inherited.",
    "because I lost my wallet yesterday.", "for an electric vehicle.",
    "in relation to my motorhome.", "for a quad bike.", "because my doctor advised me to."
]

# DVSA (Testing & Standards)
DVSA_INTENTS = [
    "I need to book my practical driving test", "How do I cancel my driving theory test", 
    "I want to check the MOT history of a vehicle", "I need to update my ADI registration details", 
    "How do we apply to become an authorized MOT testing station", "I need to report a driving instructor for bad behavior", 
    "What are the MOT testing standards", "I need to book an Individual Vehicle Approval (IVA) test", 
    "Can I change the date of my motorcycle practical test", "Where is my nearest theory test center", 
    "When can I rebook my practical exam after failing", "How do I apply for the ADI part 1 exam", 
    "Can I appeal a failed MOT decision", "I need to check if my driving instructor is actually approved",
    "How do I book an extended driving test after a ban", "What are the rules for HGV weight limits",
    "How do I get a replacement ADI badge", "Can I take my practical test in my own car",
    "How do I report an illegally operating MOT garage", "I need to book a Motorcycle Single Vehicle Approval (MSVA)"
]
DVSA_CONTEXTS = [
    "for a kit car I just built.", "for my heavy goods vehicle (HGV).", "because I failed it yesterday.",
    "for my Approved Driving Instructor (ADI) certification.", "regarding a commercial trailer.",
    "because my previous instructor was unqualified.", "for a custom imported truck.",
    "since I have a medical condition requiring an adapted car.", "because the garage failed my vehicle.",
    "for a local testing center.", "since I am completely deaf and need an interpreter.",
    "for a fast-track cancellation slot.", "because the examiner was incredibly rude.",
    "regarding my fleet of commercial buses.", "for a newly built custom motorcycle.",
    "because my MOT expires next week.", "for a dual-control teaching vehicle.",
    "since my driving ban has officially ended.", "for a newly established garage business.",
    "because I need it for my job application."
]

# NEAR OOD (Adversarial Trap Actions)
NEAR_OOD_ACTIONS = [
    "I need to appeal an official billing fine", "How do I submit a civil compensation claim to my local council", 
    "I want to file a commercial dispute with Trading Standards", "Where do I file a formal theft report with the police", 
    "Can you provide a private insurance quote", "I need to challenge a late-payment invoice sent by a contractor", 
    "How do I request a certified document translation from a notary", "I want to contest a penalty charge notice issued by the court", 
    "Where do I report an unsafe accident to my corporate fleet manager", "How do I apply for a local authority street parking permit",
    "I need to hire a private solicitor", "How do I get a loan from the bank",
    "Where can I buy aftermarket replacement parts", "I want to complain to the national ombudsman",
    "Can you help me fill out a small claims court form", "How do I scrap a vehicle at a private scrapyard",
    "I need to contact a vehicle recovery towing company", "Where do I get a local resident toll discount",
    "How do I write a legal will", "I need to find a private driving school business"
]
NEAR_OOD_TRAPS = [
    "regarding a vehicle excise duty tax problem", "involving my penalty points endorsement",
    "regarding my V5C logbook", "concerning a medical fitness to drive declaration",
    "involving a statutory off-road notification (SORN)", "regarding my practical driving test",
    "involving a certified MOT garage station", "regarding a Heavy Goods Vehicle (HGV) weight limit",
    "for an Individual Vehicle Approval (IVA) custom kit", "involving an Approved Driving Instructor (ADI)"
]
NEAR_OOD_CONTEXTS = [
    "that occurred last week.", "which is causing me a lot of financial stress.",
    "for my commercial delivery van.", "that happened in central London.",
    "because the local authority ignored me.", "for a high-emissions hybrid vehicle.",
    "that involves a foreign national.", "because I refuse to pay the fee.",
    "which was heavily damaged in a crash.", "that my employer is asking about.",
    "because my lawyer told me to.", "which is completely unacceptable.",
    "that I received in the mail yesterday.", "for my dual-control instructor vehicle.",
    "because it is currently immobilized.", "that took place in a public parking lot.",
    "which involves a third-party debt collector.", "for my imported vintage motorcycle.",
    "because it was unfairly issued.", "that I desperately need resolved."
]

# FAR OOD (Completely Irrelevant)
FAR_OOD_INTENTS = [
    "How do I bake a chocolate cake", "What time does the supermarket close today",
    "Will it rain tomorrow in London", "What is the capital of France",
    "How do I fix a leaking pipe in my kitchen sink", "Can you help me with my math homework",
    "What are the best movies playing in theaters right now", "How do I train my dog to fetch",
    "What is the current stock price of Apple", "How do I reset my home wifi router",
    "Can you recommend a good Italian restaurant", "What are the symptoms of a common cold",
    "How do I apply for a passport", "What is the history of the Roman Empire",
    "How do I build a wooden birdhouse", "Can you translate this sentence into Spanish",
    "What are the rules of basketball", "How do I knit a sweater",
    "When is the next solar eclipse", "What is the best way to clean a carpet"
]
FAR_OOD_CONTEXTS = [
    "for my daughter's birthday.", "because I have a test tomorrow.",
    "since I am planning a vacation.", "for a school project.",
    "because I am incredibly bored.", "since it has been bothering me all day.",
    "for my weekend plans.", "because my friends are coming over.",
    "since I want to learn a new skill.", "for a completely unrelated task.",
    "because I saw it on the television.", "since I am redecorating my house.",
    "for my grandmother.", "because I want to start a new hobby.",
    "since I am writing a book about it.", "for my upcoming university exams.",
    "because I have a bet with my brother.", "since I am cooking dinner tonight.",
    "for my outdoor garden.", "because I am curious."
]

aws_retry_config = Config(retries={'max_attempts': 10, 'mode': 'adaptive'})

class UnifiedCorpusGenerator:
    def __init__(self, region_name='eu-west-2'):
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region_name, config=aws_retry_config)
        self.models = [
            "eu.anthropic.claude-haiku-4-5-20251001-v1:0", 
            "anthropic.claude-sonnet-4-6",            
            "amazon.nova-pro-v1:0"           
        ]
        self.personas = [
            "a stressed citizen typing hastily on a mobile phone with minor typos.",
            "an elderly person who is very polite and slightly confused.",
            "a highly formal, almost legal-sounding written request.",
            "a frustrated user who uses a lot of slang.",
            "a non-native English speaker using simplified vocabulary."
        ]

    def get_clean_pool(self, openings, actions, contexts, label, prefix, sample_size, use_traps=False, traps=None):
        """Generates clean combinations and attaches the necessary base_intent_id tags."""
        if use_traps and traps:
            all_combos = list(itertools.product(openings, actions, traps, contexts))
            selected_combos = random.sample(all_combos, sample_size)
        else:
            all_combos = list(itertools.product(openings, actions, contexts))
            selected_combos = random.sample(all_combos, sample_size)
            
        clean_dataset = []
        for counter, combo in enumerate(selected_combos, start=1):
            if use_traps:
                op, action, trap, ctx = combo
                group_id_index = traps.index(trap)
                full_intent_text = f"{action} {trap}"
            else:
                op, action, ctx = combo
                group_id_index = actions.index(action)
                full_intent_text = action

            if random.random() > 0.5:
                dialogue = [f"{op}{full_intent_text} {ctx}"]
            else:
                dialogue = [f"{op}{full_intent_text}.", f"It is {ctx}"]
                
            clean_dataset.append({
                "id": f"{prefix}_{counter:04d}",
                "base_intent_id": f"{prefix}_group_{group_id_index}",
                "ground_truth": label,
                "dialogue": dialogue
            })
        return clean_dataset

    def _paraphrase_text(self, original_text, max_retries=5):
        """Calls AWS Bedrock to apply adversarial noise."""
        model_id = random.choice(self.models)
        persona = random.choice(self.personas)
        system_instruction = f"Rewrite the following user utterance as if it were written by {persona}. Output ONLY the paraphrased query with no introductory text."
        
        for attempt in range(max_retries):
            try:
                response = self.bedrock_client.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": original_text}]}],
                    system=[{"text": system_instruction}],
                    inferenceConfig={"temperature": 0.8, "maxTokens": 150}
                )
                return response['output']['message']['content'][0]['text'].strip()
            except ClientError:
                time.sleep(2 ** attempt)
        return original_text 

    def _process_item(self, item):
        """Paraphrases all turns in a single trajectory while preserving IDs."""
        new_dialogue = [self._paraphrase_text(turn) for turn in item["dialogue"]]
        return {
            "id": item["id"],
            "base_intent_id": item["base_intent_id"],
            "ground_truth": item["ground_truth"],
            "dialogue": new_dialogue
        }

    def generate_paraphrased_corpus(self, clean_corpus):
        """Processes the clean corpus through LLMs concurrently to create the Phase II dataset."""
        paraphrased_corpus = []
        with ThreadPoolExecutor(max_workers=25) as executor:
            futures = [executor.submit(self._process_item, item) for item in clean_corpus]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Phase II (Paraphrasing)"):
                paraphrased_corpus.append(future.result())
        return paraphrased_corpus


def main():
    os.makedirs("data", exist_ok=True)
    random.seed(42)  
    generator = UnifiedCorpusGenerator()
    
    print("\n" + "="*60)
    print("STEP 1: GENERATING PHASE I (CLEAN BASELINE)")
    print("="*60)
    TARGET_PER_CLASS = 2500
    
    dvla_data = generator.get_clean_pool(OPENINGS, DVLA_INTENTS, DVLA_CONTEXTS, "DVLA", "traj_dvla", TARGET_PER_CLASS)
    dvsa_data = generator.get_clean_pool(OPENINGS, DVSA_INTENTS, DVSA_CONTEXTS, "DVSA", "traj_dvsa", TARGET_PER_CLASS)
    near_ood_data = generator.get_clean_pool(OPENINGS, NEAR_OOD_ACTIONS, NEAR_OOD_CONTEXTS, "OOD", "traj_ood_near", TARGET_PER_CLASS, use_traps=True, traps=NEAR_OOD_TRAPS)
    far_ood_data = generator.get_clean_pool(OPENINGS, FAR_OOD_INTENTS, FAR_OOD_CONTEXTS, "OOD", "traj_ood_far", TARGET_PER_CLASS)
    
    phase1_dataset = dvla_data + dvsa_data + near_ood_data + far_ood_data
    random.shuffle(phase1_dataset)
    
    p1_path = os.path.join("data", "synthetic_corpus_phase1.json")
    with open(p1_path, "w", encoding="utf-8") as f:
        json.dump(phase1_dataset, f, indent=4)
    print(f"Success: Saved 10,000 clean trajectories to '{p1_path}'\n")

    print("="*60)
    print("STEP 2: GENERATING PHASE II (ADVERSARIAL DEGRADATION)")
    print("="*60)
    
    phase2_dataset = generator.generate_paraphrased_corpus(phase1_dataset)
    
    p2_path = os.path.join("data", "synthetic_corpus_phase2.json")
    with open(p2_path, "w", encoding="utf-8") as f:
        json.dump(phase2_dataset, f, indent=4)
    print(f"\nSuccess: Saved 10,000 paraphrased trajectories to '{p2_path}'")
    print("Unified Generation Complete. You may now run main.py.")

if __name__ == "__main__":
    main()