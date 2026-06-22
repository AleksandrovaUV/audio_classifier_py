import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import hdbscan
from typing import List, Dict
import re


class TopicAnalyzer:
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        small_dialog_threshold: int = 5,
        distance_threshold: float = 0.65,
        min_cluster_size: int = 2,
    ):
        self.model = SentenceTransformer(embedding_model)
        self.small_dialog_threshold = small_dialog_threshold
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size

    # -----------------------------
    # 1. keywords extraction
    # -----------------------------


    def extract_keywords(self, text: str, top_k: int = 3):
        text_low = text.lower()

        # model = re.findall(r"(iphone\s*\d{1,3})", text_low)
        
        # memory = re.findall(r"(\d+\s*гб|\d+\s*gb)", text_low)

        # numbers = re.findall(r"\b\d{2,4}\b", text_low)

        # brands = re.findall(r"(iphone|samsung|xiaomi|huawei|nokia|sony)", text_low)

        # entities = model + memory + brands + numbers

        # if entities:
        #     seen = set()
        #     uniq = []
        #     for e in entities:
        #         if e not in seen:
        #             uniq.append(e)
        #             seen.add(e)
        #     return uniq[:top_k]

        # fallback — обычные ключевые слова
        words = [
            w.strip(".,!?;:()[]«»").lower()
            for w in text.split()
            if len(w) > 4
        ]
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:top_k]] or ["topic"]


    # -----------------------------
    # 2. Рsmall topic separation
    # -----------------------------

    def split_small_dialog(self, segments, embeddings):
        sims = cosine_similarity(embeddings)
        topics = []
        current = [segments[0]]

        for i in range(1, len(segments)):
            sim = sims[i - 1][i]
            if sim < self.distance_threshold:
                topics.append(current)
                current = []
            current.append(segments[i])

        topics.append(current)
        return topics

    # -----------------------------
    # 3. big topic separation
    # -----------------------------
    def cluster_large_dialog(self, segments, embeddings):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean"
        )
        labels = clusterer.fit_predict(embeddings)

        clusters = {}
        for seg, label in zip(segments, labels):
            clusters.setdefault(label, []).append(seg)

        return clusters

    # -----------------------------
    # 4. topic name generation
    # -----------------------------
    def make_topic_name(self, segments):
        text = " ".join([s["text"] for s in segments])
        keywords = self.extract_keywords(text)
        return " / ".join(keywords)

    # -----------------------------
    # 5. main
    # -----------------------------
    def analyze(self, segments: List[Dict]):
        texts = [s["text"] for s in segments]
        embeddings = self.model.encode(texts)

        # small dialigue -- simple extraction
        if len(segments) < self.small_dialog_threshold:
            topic_groups = self.split_small_dialog(segments, embeddings)
            topics = []
            for i, group in enumerate(topic_groups):
                topics.append({
                    "topic_id": i,
                    "topic_name": self.make_topic_name(group),
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "segments": group
                })
            return topics

        # long dialog -- HDBSCAN
        clusters = self.cluster_large_dialog(segments, embeddings)
        topics = []
        for label, group in clusters.items():
            topics.append({
                "topic_id": int(label),
                "topic_name": self.make_topic_name(group),
                "start": group[0]["start"],
                "end": group[-1]["end"],
                "segments": group
            })

        return topics

