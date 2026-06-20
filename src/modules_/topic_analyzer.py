from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import hdbscan
from collections import defaultdict
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TopicAnalyzer:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_segments(self, segments: List[Dict[str, Any]]) -> np.ndarray:
        texts = [seg["text"] for seg in segments]
        logger.info(f"Embedding {len(texts)} segments")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings

    def cluster_embeddings(self, embeddings: np.ndarray) -> List[int]:
        logger.info("Clustering embeddings with HDBSCAN")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        labels = clusterer.fit_predict(embeddings)
        return labels

    def summarize_topic(self, texts: List[str]) -> str:
        from collections import Counter
        words = " ".join(texts).lower().split()
        common = Counter(words).most_common(5)
        top_words = [w for w, _ in common]
        return " / ".join(top_words)

    def analyze(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not segments:
            return []

        embeddings = self.embed_segments(segments)
        labels = self.cluster_embeddings(embeddings)

        # Группируем сегменты по темам
        topic_groups = defaultdict(list)
        for seg, label in zip(segments, labels):
            topic_groups[label].append(seg)

        topics = []
        for label, segs in topic_groups.items():
            if label == -1:
                topic_name = "Miscellaneous"
            else:
                topic_name = self.summarize_topic([s["text"] for s in segs])

            start = segs[0]["start"]
            end = segs[-1]["end"]

            topics.append({
                "topic_id": int(label),
                "topic_name": topic_name,
                "start": start,
                "end": end,
                "segments": segs
            })

        logger.info(f"Detected {len(topics)} topics")
        return topics
