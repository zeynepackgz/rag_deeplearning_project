import os
import yaml
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2


class KnowledgeEnhancer:
    def __init__(self):
        self.config = self._load_config()
        self.model = SentenceTransformer(self.config['model_settings']['embedding_model'])
        self.knowledge = self._build_knowledge_base()

    def _load_config(self):
        config_path = Path(__file__).parent.parent / "config" / "rag_config.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _process_file(self, file_path):
        text = ""
        if file_path.suffix == ".pdf":
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join([page.extract_text() for page in reader.pages])
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        return text.split('\n')

    def _build_knowledge_base(self):
        kb_path = Path(self.config['knowledge_base']['path'])
        documents = []

        # Dosyalardan veri çek
        for file in kb_path.glob('*'):
            if file.suffix in self.config['knowledge_base']['allowed_extensions']:
                documents.extend(self._process_file(file))

        # Varsayılan veriler
        if not documents:
            documents = self.config['knowledge_base']['default_responses']

        # Embedding'leri oluştur
        embeddings = self.model.encode(documents)
        return {
            'documents': documents,
            'embeddings': embeddings
        }

    def find_similar(self, query, top_k=3):
        query_embed = self.model.encode([query])
        similarities = cosine_similarity(query_embed, self.knowledge['embeddings'])
        sorted_indices = np.argsort(similarities[0])[::-1][:top_k]

        results = []
        for idx in sorted_indices:
            if similarities[0][idx] > self.config['model_settings']['similarity_threshold']:
                results.append({
                    'text': self.knowledge['documents'][idx],
                    'score': float(similarities[0][idx])
                })
        return results

    def enhance_app_response(self, apps, query):
        app_embeds = self.model.encode([app['name'] + " " + app.get('description', '') for app in apps])
        query_embed = self.model.encode([query])[0]

        scores = cosine_similarity([query_embed], app_embeds)[0]
        best_idx = np.argmax(scores)

        if scores[best_idx] > 0.5:  # Uygulama spesifik eşik
            return apps[best_idx]
        return None