# app.py - API Flask pour RAFT Agent
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

app = Flask(__name__)
CORS(app)  # Permettre les requêtes depuis Flutter

print("🔥 Chargement du modèle RAFT...")

# Charger le modèle et tokenizer
MODEL_PATH = "./raft_model_final"
try:
    model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={"": "cpu"},  # ✅ Force CPU uniquement
    torch_dtype=torch.float32,
    trust_remote_code=True,
    low_cpu_mem_usage=True  # ✅ Optimise l'utilisation mémoire
)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("✅ Modèle RAFT chargé avec succès!")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    model = None
    tokenizer = None

def raft_answer(question, context):
    """Fonction d'inférence RAFT"""
    if model is None or tokenizer is None:
        return "Erreur: Modèle non chargé"
    
    # Créer le prompt
    prompt = f"""Context: {context[:500]}

Question: {question}

Answer:"""
    
    # Tokenization
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=768,
        truncation=True
    )
    
    # Génération
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.35,
            top_p=0.85,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=3
        )
    
    # Décodage
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraction de la réponse
    if "Answer:" in response:
        answer = response.split("Answer:")[-1].strip()
    else:
        answer = response.strip()
    
    # Nettoyage
    answer = answer.split("\n")[0].strip()
    answer = answer.split("Context:")[0].strip()
    answer = answer.split("Question:")[0].strip()
    
    return answer

# ============================================================
# ROUTES API
# ============================================================

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil de l'API"""
    return jsonify({
        "status": "ok",
        "message": "RAFT Agriculture API is running",
        "model": "TinyLlama-1.1B-RAFT",
        "endpoints": {
            "/health": "Health check",
            "/raft": "POST - RAFT inference"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })

@app.route('/raft', methods=['POST'])
def raft_endpoint():
    """
    Endpoint principal pour RAFT
    
    Request JSON:
    {
        "question": "What are symptoms of gray leaf spot?",
        "context": "Gray leaf spot causes rectangular lesions..."
    }
    """
    try:
        # Récupérer les données
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        question = data.get("question", "").strip()
        context = data.get("context", "").strip()
        
        # Validation
        if not question:
            return jsonify({"error": "Question is required"}), 400
        
        if not context:
            return jsonify({"error": "Context is required"}), 400
        
        # Inférence
        print(f"📥 Question: {question[:50]}...")
        answer = raft_answer(question, context)
        print(f"📤 Answer: {answer[:100]}...")
        
        # Réponse
        return jsonify({
            "status": "success",
            "question": question,
            "answer": answer
        })
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ============================================================
# DÉMARRAGE
# ============================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*60}")
    print("🌾 RAFT Agriculture API")
    print(f"{'='*60}")
    print(f"🚀 Server running on http://0.0.0.0:{port}")
    print(f"📡 Endpoints:")
    print(f"   GET  / - API info")
    print(f"   GET  /health - Health check")
    print(f"   POST /raft - RAFT inference")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)