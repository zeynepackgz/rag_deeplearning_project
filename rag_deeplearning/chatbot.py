import random
import json
import numpy as np
from typing import Optional
from modules.knowledge_enhancer import KnowledgeEnhancer
from nltk.stem import WordNetLemmatizer
import pickle
from tensorflow.keras.models import load_model
import nltk

class ChatBot:
    def __init__(self, intents_path="intents.json"):
        self.intents = self._load_intents(intents_path)
        self.lemmatizer = WordNetLemmatizer()
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
        self.model = load_model('chatbot_model.h5')

    def _load_intents(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # Mevcut DL model fonksiyonlarınız
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def process_message(self, message: str, knowledge_enhancer: Optional[KnowledgeEnhancer] = None):
        # Intent tahmini
        ints = self.predict_class(message)

        # Intent işleme
        try:
            tag = ints[0]['intent']
            probability = float(ints[0]['probability'])

            # Yüksek güvenilirlik durumu
            if probability > 0.7:
                return self._get_intent_response(tag)

            # RAG ile geliştirilmiş yanıt
            if knowledge_enhancer:
                # Uygulama spesifik geliştirme
                if tag in ['sağlık_mobil_uygulamaları', 'şirket_mobil_uygulamaları']:
                    enhanced = self._enhance_app_response(tag, message, knowledge_enhancer)
                    if enhanced:
                        return enhanced

                # Genel bilgi tabanı
                rag_response = knowledge_enhancer.find_similar(message)
                if rag_response:
                    return self._format_rag_response(rag_response)

            # Fallback
            return {"type": "text", "content": "Anlayamadım, lütfen farklı şekilde ifade edin."}

        except Exception as e:
            print(f"Hata: {str(e)}")
            return {"type": "error", "content": "Bir hata oluştu"}

    def _get_intent_response(self, tag):
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                return self._format_response(response)

    def _enhance_app_response(self, tag, query, knowledge_enhancer):
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                apps = next((r['apps'] for r in intent['responses'] if 'apps' in r), [])
                best_app = knowledge_enhancer.enhance_app_response(apps, query)
                if best_app:
                    return {
                        "type": "enhanced_app",
                        "app": best_app['name'],
                        "description": best_app.get('description', ''),
                        "url": best_app['url']
                    }
        return None

    def _format_response(self, response):
        if isinstance(response, dict):
            return {
                "type": "apps" if 'apps' in response else "links",
                "content": response
            }
        return {"type": "text", "content": response}

    def _format_rag_response(self, results):
        return {
            "type": "knowledge",
            "content": {
                "header": "İlgili Bilgiler:",
                "items": [res['text'] for res in results[:3]]
            }
        }