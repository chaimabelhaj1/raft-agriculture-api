
╔══════════════════════════════════════════════════════════╗
║         MODÈLE RAFT - AGRICULTURE INTELLIGENTE          ║
╚══════════════════════════════════════════════════════════╝

🚀 UTILISATION
────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained("./raft_model_final")
tokenizer = AutoTokenizer.from_pretrained("./raft_model_final")

def raft_answer(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=768, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, 
                                temperature=0.35, repetition_penalty=1.2)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

📱 API FLASK
────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify

app = Flask(__name__)
model = AutoModelForCausalLM.from_pretrained("./raft_model_final")
tokenizer = AutoTokenizer.from_pretrained("./raft_model_final")

@app.route("/raft", methods=["POST"])
def raft():
    data = request.json
    answer = raft_answer(data["question"], data["context"])
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

🌐 DÉPLOIEMENT RAILWAY (GRATUIT)
────────────────────────────────────────────────────────────
1. railway.app → New Project
2. requirements.txt:
   transformers==4.38.0
   torch==2.1.0
   flask==3.0.0
   peft==0.8.2
3. Deploy!

✨ Bon développement! 🌾
