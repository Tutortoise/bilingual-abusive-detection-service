import tensorflow as tf
import pickle
from tensorflow.keras.layers import TextVectorization, Layer
from tensorflow.keras import layers
from keras.saving import register_keras_serializable
from .config import EN_ABUSIVE_PATH, ID_ABUSIVE_PATH
from .exceptions import EmptyTextError
from .words import AbusiveWordsDetector


@register_keras_serializable()
class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embeddings = None
        self.position_embeddings = None

    def build(self, input_shape):
        self.token_embeddings = layers.Embedding(
            self.vocab_size, self.embed_dim, name="token_embedding"
        )
        self.position_embeddings = layers.Embedding(
            self.sequence_length, self.embed_dim, name="position_embedding"
        )
        super().build(input_shape)

    def call(self, inputs):
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


@register_keras_serializable()
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        # Initialize as None
        self.att = None
        self.ffn = None
        self.layernorm1 = None
        self.layernorm2 = None
        self.dropout1 = None
        self.dropout2 = None

    def build(self, input_shape):
        self.att = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            name="multi_head_attention",
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(
                    self.ff_dim,
                    activation="gelu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    name="ff_dense_1",
                ),
                layers.Dropout(self.rate, name="ff_dropout"),
                layers.Dense(
                    self.embed_dim,
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                    name="ff_dense_2",
                ),
            ],
            name="feed_forward_network",
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6, name="layernorm_1")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6, name="layernorm_2")
        self.dropout1 = layers.Dropout(self.rate, name="dropout_1")
        self.dropout2 = layers.Dropout(self.rate, name="dropout_2")

        super().build(input_shape)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
            }
        )
        return config


class AbusiveTextDetector:
    def __init__(self, model_path, vectorizer_config_path, threshold=0.7):
        self.threshold = threshold
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "PositionalEmbedding": PositionalEmbedding,
                "TransformerBlock": TransformerBlock,
            },
        )

        with open(vectorizer_config_path, "rb") as f:
            self.vectorizer_config = pickle.load(f)

        self.vectorizer = TextVectorization(
            max_tokens=self.vectorizer_config["max_tokens"],
            output_sequence_length=self.vectorizer_config["sequence_length"],
            output_mode="int",
        )
        self.vectorizer.set_vocabulary(self.vectorizer_config["vocabulary"])

        self.abusive_detector = AbusiveWordsDetector(EN_ABUSIVE_PATH, ID_ABUSIVE_PATH)

    def preprocess_text(self, text):
        text_tensor = tf.convert_to_tensor([text])
        vectorized_text = self.vectorizer(text_tensor)
        return vectorized_text

    def get_prediction_label(self, probability):
        """Determine if text is abusive based on probability and confidence threshold"""
        confidence = float(abs(probability - 0.5) * 2)
        is_abusive = probability > 0.5

        if is_abusive and confidence < self.threshold:
            is_abusive = False

        return {
            "probability": float(probability),
            "is_abusive": is_abusive,
            "confidence": confidence,
            "early_detection": False,
            "matched_words": [],
        }

    def validate_text(self, text):
        """Validate input text"""
        if not text or not text.strip():
            raise EmptyTextError("Input text cannot be empty")

    def validate_texts(self, texts):
        """Validate batch input texts"""
        if not texts:
            raise EmptyTextError("Input texts list cannot be empty")
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise EmptyTextError(f"Text at index {i} cannot be empty")

    def predict(self, text):
        self.validate_text(text)

        # First, check for abusive words
        early_detection = self.abusive_detector.contains_abusive_words(text)
        if early_detection["is_abusive"]:
            return {
                "text": text,
                "probability": 1.0,
                "is_abusive": True,
                "confidence": 1.0,
                "matched_words": early_detection["matched_words"],
                "early_detection": True,
            }

        # If no early detection, use the model
        vectorized_text = self.preprocess_text(text)
        prediction = self.model.predict(vectorized_text, verbose=0)[0][0]

        result = self.get_prediction_label(prediction)
        result["text"] = text

        return result

    def predict_batch(self, texts):
        self.validate_texts(texts)

        results = []
        early_detection_indices = set()

        # First pass: check for abusive words
        for i, text in enumerate(texts):
            early_detection = self.abusive_detector.contains_abusive_words(text)
            if early_detection["is_abusive"]:
                results.append(
                    {
                        "text": text,
                        "probability": 1.0,
                        "is_abusive": True,
                        "confidence": 1.0,
                        "matched_words": early_detection["matched_words"],
                        "early_detection": True,
                    }
                )
                early_detection_indices.add(i)

        # Second pass: process remaining texts with the model
        remaining_texts = [
            text for i, text in enumerate(texts) if i not in early_detection_indices
        ]

        if remaining_texts:
            text_tensor = tf.convert_to_tensor(remaining_texts)
            vectorized_texts = self.vectorizer(text_tensor)
            predictions = self.model.predict(vectorized_texts, verbose=0)

            for text, pred in zip(remaining_texts, predictions):
                result = self.get_prediction_label(pred[0])
                result["text"] = text
                results.append(result)

        # Sort results back to original order
        sorted_results = []
        result_idx = 0
        for i in range(len(texts)):
            if i in early_detection_indices:
                sorted_results.append(next(r for r in results if r["text"] == texts[i]))
            else:
                sorted_results.append(results[result_idx])
                result_idx += 1

        return sorted_results
