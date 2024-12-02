from typing import Set, Dict, List
import csv
import re
from collections import defaultdict


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_end = False
        self.original_word = None


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, original: str = None):
        node = self.root
        for char in word:
            node = node.children[char]
        node.is_end = True
        node.original_word = original or word

    def search(self, word: str) -> str:
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.original_word if node.is_end else None


class TextSanitizer:
    def __init__(self):
        self.patterns = {
            "repeating": re.compile(r"(.)\1{2,}"),
            "non_alpha": re.compile(r"[^a-zA-Z\s]"),
            "whitespace": re.compile(r"\s+"),
            "numbers": re.compile(r"\d+"),
        }

        self.char_map = str.maketrans(
            {
                "0": "o",
                "1": "i",
                "2": "z",
                "3": "e",
                "4": "a",
                "5": "s",
                "6": "g",
                "7": "t",
                "8": "b",
                "9": "g",
                "@": "a",
                "$": "s",
                "!": "i",
                "&": "n",
                "'": "",
                '"': "",
                "`": "",
                ".": "",
                ",": "",
                "-": "",
                "_": "",
                "=": "",
                "/": "",
                "\\": "",
                "|": "",
                "<": "",
                ">": "",
                "(": "",
                ")": "",
                "[": "",
                "]": "",
                "{": "",
                "}": "",
                "~": "",
                "^": "",
                "*": "",
                "+": "",
                ":": "",
                ";": "",
                "?": "",
                "¿": "",
                "!": "",
                "¡": "",
            }
        )

    def _split_compound_words(self, text: str) -> List[str]:
        """Split potential compound words into individual words"""
        words = []
        for word in text.split():
            current_word = ""
            for char in word:
                current_word += char
                if len(current_word) >= 3:  # Minimum word length
                    words.append(current_word)
            if current_word:
                words.append(current_word)
        return words

    def _normalize_repeating(self, word: str) -> str:
        """Normalize repeating characters"""
        result = self.patterns["repeating"].sub(r"\1\1", word)
        return result

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = text.translate(self.char_map)
        words = text.split()
        cleaned_words = []

        for word in words:
            word = self.patterns["non_alpha"].sub("", word)
            if word:
                cleaned_words.append(word)

        seen = set()
        unique_words = []
        for word in cleaned_words:
            if word not in seen:
                seen.add(word)
                unique_words.append(word)

        return " ".join(unique_words).strip()


class AbusiveWordsDetector:
    def __init__(self, en_abusive_path: str, id_abusive_path: str):
        self.sanitizer = TextSanitizer()
        self.en_trie = Trie()
        self.id_trie = Trie()

        self.en_words = self._load_abusive_words(en_abusive_path, self.en_trie)
        self.id_words = self._load_abusive_words(id_abusive_path, self.id_trie)

        self.clean_cache = {}

    def _load_abusive_words(self, file_path: str, trie: Trie) -> Set[str]:
        loaded_words = set()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                csv_reader = csv.reader(f)
                _ = next(csv_reader)  # Skip header

                for row in csv_reader:
                    if row and (word := row[0].strip()):
                        cleaned = self.sanitizer.clean_text(word)
                        if cleaned:
                            loaded_words.add(cleaned)
                            trie.insert(cleaned, cleaned)
                            self._add_variations(cleaned, trie)
        except Exception as e:
            print(f"Error loading abusive words from {file_path}: {str(e)}")
            raise

        return loaded_words

    def _add_variations(self, word: str, trie: Trie):
        variations = set()
        variations.add(word)

        pattern = re.compile(r"(.)\1+")
        base_word = pattern.sub(r"\1", word)

        for i, char in enumerate(base_word):
            if char in "aios":
                variation = list(base_word)
                variation[i] = {"a": "4", "i": "1", "o": "0", "s": "5"}[char]
                variations.add("".join(variation))

        for variation in variations:
            trie.insert(variation, word)

    def contains_abusive_words(self, text: str) -> Dict[str, any]:
        cache_key = text
        if cache_key in self.clean_cache:
            return self.clean_cache[cache_key]

        cleaned_text = self.sanitizer.clean_text(text)
        words = cleaned_text.split()

        matches = set()
        for word in words:
            pattern = re.compile(r"(.)\1+")
            normalized_word = pattern.sub(r"\1", word)

            if en_match := self.en_trie.search(normalized_word):
                matches.add(en_match)
            if id_match := self.id_trie.search(normalized_word):
                matches.add(id_match)

        found_abusive = bool(matches)
        result = {
            "is_abusive": found_abusive,
            "confidence": 1.0 if found_abusive else 0.0,
            "matched_words": list(matches) if found_abusive else [],
            "early_detection": True,
            "cleaned_text": cleaned_text,
        }

        self.clean_cache[cache_key] = result
        return result
