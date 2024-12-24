# FILE: finsearch/retrieval/model/bm25.py

import json
from typing import List

import numpy as np
import torch
from openai import BadRequestError, OpenAI
from torch import cosine_similarity
from finsearch.retrieval.interface import IRetrieval
from finsearch.retrieval.config import BM25Config
from finsearch.util import download_data

import os
import pickle
import contextlib
import heapq
import math
import re
from porter2stemmer import Porter2Stemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import string
import tiktoken
import xgboost as xgb

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from trie import Trie
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    trie(Trie): Class Trie untuk query auto-completion
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.trie = Trie()

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

        self.stemmer = Porter2Stemmer()
        self.stop_words = set(stopwords.words('english'))

    def save(self):
        """Menyimpan doc_id_map, term_id_map, dan trie ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        with open(os.path.join(self.output_dir, 'trie.pkl'), 'wb') as f:
            # file ini mungkin agak besar
            pickle.dump(self.trie, f)

    def load(self):
        """Memuat doc_id_map, term_id_map, dan trie dari output directory"""
        terms_dict_path = os.path.join(self.output_dir, 'terms.dict')
        print(f"Looking for terms.dict at: {terms_dict_path} from {os.getcwd()}")

        if not os.path.exists(terms_dict_path):
            raise FileNotFoundError(f"File not found: {terms_dict_path}")
        
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'trie.pkl'), 'rb') as f:
            self.trie = pickle.load(f)

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Anda bisa menggunakan stemmer bahasa Inggris yang tersedia, seperti Porter Stemmer
        https://github.com/evandempsey/porter2-stemmer

        Untuk membuang stopwords, Anda dapat menggunakan library seperti NLTK.

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # TODO
        # Hint: Anda dapat mengisi trie di sini
        td_pairs = []
        term_freq = {}
        block_folder_path = os.path.join(self.data_dir, block_path)

        for file_name in os.listdir(block_folder_path):
            file_path = os.path.join(block_folder_path, file_name)

            doc_id = self.doc_id_map[file_path]

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                terms = re.findall(r'\b\w+\b', content.lower())
                for raw_term in terms:
                    if raw_term in self.stop_words:
                        continue
                    
                    processed_term = self.stemmer.stem(raw_term)
                    term_id = self.term_id_map[processed_term]
                    td_pairs.append((term_id, doc_id))
                    
                    term_freq[raw_term] = term_freq.get(raw_term, 0) + 1
    
        for raw_term, freq in term_freq.items():
            self.trie.insert(raw_term, freq)

        return td_pairs


    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # term_dict merupakan dictionary yang berisi dictionary yang
        # melakukan mapping dari doc_id ke tf
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = dict()
            # Mengupdate juga TF (yang merupakan value dari dictionary yang di dalam)
            term_dict[term_id][doc_id] = term_dict[term_id].get(doc_id, 0) + 1
        
        for term_id in sorted(term_dict.keys()):
            # Sort postings list (dan tf list yang bersesuaian)
            sorted_postings_tf = dict(sorted(term_dict[term_id].items()))
            # Postings list adalah keys, TF list adalah values
            index.append(term_id, list(sorted_postings_tf.keys()), 
                         list(sorted_postings_tf.values()))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def _compute_score_tfidf(self, tf, df, N):
        """
        Fungsi ini melakukan komputasi skor TF-IDF.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        score = w(t, Q) x w(t, D)
        Tidak perlu lakukan normalisasi pada score.

        Gunakan log basis 10.

        Parameters
        ----------
        tf: int
            Term frequency.

        df: int
            Document frequency.

        N: int
            Jumlah dokumen di corpus. 

        Returns
        -------
        float
            Skor hasil perhitungan TF-IDF.
        """
        # TODO
        wtd = (1 + math.log10(tf)) if tf > 0 else 0
        idf = math.log(N/df)
        return wtd * idf
    
    def _compute_score_bm25(self, tf, df, N, k1, b, dl, avdl):
        """
        Fungsi ini melakukan komputasi skor BM25.
        Gunakan log basis 10 dan tidak perlu lakukan normalisasi.
        Silakan lihat penjelasan parameters di slide.

        Returns
        -------
        float
            Skor hasil perhitungan BM25.
        """
        # TODO
        idf = math.log(N/df)
        adjusted_tf = ((k1 + 1) * tf) / (k1 * ((1 - b) + b * (dl / avdl)) + tf)
        return idf * adjusted_tf
    
    def _preprocess_query(self, query):
        tokenized_query = re.findall(r'\b\w+\b', query.lower())
        tokenized_query = [self.stemmer.stem(term).lower() for term in tokenized_query if term not in self.stop_words]
        return tokenized_query

    def retrieve_tfidf_daat(self, query, k=10):
        """
        Lakukan retrieval TF-IDF dengan skema DaaT.
        Method akan mengembalikan top-K retrieval results.

        Program tidak perlu paralel sepenuhnya. Untuk mengecek dan mengevaluasi
        dokumen yang di-point oleh pointer pada waktu tertentu dapat dilakukan
        secara sekuensial, i.e., seperti menggunakan for loop.

        Beberapa informasi penting: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        doc_scores = {}  # Dictionary to store document scores
        query_terms = self._preprocess_query(query)

        with InvertedIndexReader(self.index_name, 
                                self.postings_encoding, 
                                directory=self.output_dir) as index:
            N = len(index.doc_length)  # Total number of documents
            postings_dict = index.postings_dict

            term_postings = []  # List to hold postings lists and related info for each term
            for term in query_terms:
                if term in self.term_id_map:
                    term_id = self.term_id_map[term]
                    if term_id in postings_dict:
                        df = postings_dict[term_id][1]  # Document frequency of the term
                        postings_list, tf_list = index.get_postings_list(term_id)
                        term_postings.append({
                            'term_id': term_id,
                            'df': df,
                            'postings_list': postings_list,
                            'tf_list': tf_list,
                            'pointer': 0
                        })

            if not term_postings:
                return []

            pointers = [0] * len(term_postings)

            while True:
                current_doc_ids = []
                for i, term_posting in enumerate(term_postings):
                    postings_list = term_posting['postings_list']
                    pointer = pointers[i]
                    if pointer < len(postings_list):
                        current_doc_ids.append(postings_list[pointer])
                if not current_doc_ids:
                    break

                current_doc_id = min(current_doc_ids)

                score = 0.0
                for i, term_posting in enumerate(term_postings):
                    postings_list = term_posting['postings_list']
                    tf_list = term_posting['tf_list']
                    df = term_posting['df']
                    pointer = pointers[i]
                    if pointer < len(postings_list) and postings_list[pointer] == current_doc_id:
                        tf = tf_list[pointer]
                        score += self._compute_score_tfidf(tf, df, N)
                        pointers[i] += 1  
                if score > 0:
                    doc_scores[current_doc_id] = score

            retrieved_docs = self.get_top_k_by_score(doc_scores, k)
            return retrieved_docs

    def retrieve_tfidf_taat(self, query, k=10):
        """
        Lakukan retrieval TF-IDF dengan skema TaaT.
        Method akan mengembalikan top-K retrieval results.

        Beberapa informasi penting: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_list
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        doc_to_sim = {}
        query_terms = self._preprocess_query(query)

        with InvertedIndexReader(self.index_name, 
                                 self.postings_encoding, 
                                 directory=self.output_dir) as index:
            N = len(index.doc_length)
            postings_dict = index.postings_dict
            for term in query_terms:
                if term not in self.term_id_map:
                    continue
                term_id = self.term_id_map[term]
    
                df = postings_dict[term_id][1]
                postings_and_tf = index.get_postings_list(term_id)

                for doc_id, tf in zip(postings_and_tf[0], postings_and_tf[1]):
                    score = self._compute_score_tfidf(tf, df, N)
                    doc_to_sim[doc_id] = doc_to_sim.get(doc_id, 0) + score

        retrieved_docs = self.get_top_k_by_score(doc_to_sim, k)
        return retrieved_docs

    def retrieve_bm25_taat(self, query, k=10, k1=1.2, b=0.75):
        """
        Lakukan retrieval BM-25 dengan skema TaaT.
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        # TODO

        doc_to_sim = {}
        query_terms = self._preprocess_query(query)

        with InvertedIndexReader(self.index_name, 
                                 self.postings_encoding, 
                                 directory=self.output_dir) as index:
            N = len(index.doc_length)
            postings_dict = index.postings_dict
            avdl = sum(index.doc_length.values()) / len(index.doc_length)
            for term in query_terms:
                if term not in self.term_id_map:
                    continue
                term_id = self.term_id_map[term]
    
                df = postings_dict[term_id][1]
                postings_and_tf = index.get_postings_list(term_id)

                for doc_id, tf in zip(postings_and_tf[0], postings_and_tf[1]):
                    dl = index.doc_length[doc_id]
                    score = self._compute_score_bm25(tf, df, N, k1, b, dl, avdl)
                    
                    doc_to_sim[doc_id] = doc_to_sim.get(doc_id, 0) + score
        
        retrieved_docs = self.get_top_k_by_score(doc_to_sim, k)
        
        return retrieved_docs


    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()
        self.intermediate_indices = ['intermediate_index_'+block_dir_relative for block_dir_relative in sorted(next(os.walk(self.data_dir))[1])]

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)

    def get_top_k_by_score(self, score_docs, k):
        """
        Method ini berfungsi untuk melakukan sorting terhadap dokumen berdasarkan score
        yang dihitung, lalu mengembalikan top-k dokumen tersebut dalam bentuk tuple
        (score, document). Silakan gunakan heap agar lebih efisien.

        Parameters
        ----------
        score_docs: Dictionary[int -> float]
            Dictionary yang berisi mapping docID ke score masing-masing dokumen tersebut.

        k: Int
            Jumlah dokumen yang ingin di-retrieve berdasarkan score-nya.

        Result
        -------
        List[(float, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.
        """
        # TODO
        top_k = heapq.nlargest(k, score_docs.items(), key=lambda x: x[1])

        result = []
        for doc_id, score in top_k:
            doc_name = self.doc_id_map[doc_id]
            result.append((score, doc_name))

        return result

    
    def get_query_recommendations(self, query, k=5):
        # Method untuk mendapatkan rekomendasi untuk QAC
        # Tidak perlu mengubah ini
        self.load()
        last_token = query.split()[-1]
        recc = self.trie.get_recommendations(last_token, k)
        return recc

class BM25Retriever():
    def __init__(self, config: BM25Config, collection: List[str]):
        self.config = config
        # ngambil param BSBIIndex
        arxiv_collections = download_data(url=self.config.arxiv_collections_url, filename=self.config.arxiv_collections_folder, dir_name=self.config.arxiv_collections_folder)
        index = download_data(url=self.config.index_url, filename=self.config.index_folder, dir_name=self.config.index_folder)
        self.collection = collection
        self.bsbi_instance = BSBIIndex(data_dir='arxiv_collections',
                                        postings_encoding=VBEPostings,
                                        output_dir='/app/index')
        self.bsbi_instance.load()

    async def retrieve(self, query: str, k: int = 1) -> List[str]:
        bm25_results = [i[-1].replace(".txt","").split("/")[-1] for i in self.bsbi_instance.retrieve_bm25_taat(query, k=30)]
        return bm25_results

class BM25RetrieverOpenAI():
    def __init__(self, base:BM25Retriever, config:BM25Config, mapper: dict):
        self.config = config
        self.base = base
        self.mapper = mapper
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        download_data(url=self.config.embed_url, filename=self.config.embed_filename)
        with open("/app/index/document_embedded.json", "r") as file:
            doc_embed_map = json.load(file)
        self.doc_vectorizer = doc_embed_map

    async def retrieve(self, query: str, k: int = 1) -> List[str]:
        bm25_results = self.base.bsbi_instance.retrieve_bm25_taat(query, k=k)
        bm25_results = self.rerank(query, bm25_results)
        return bm25_results

    def truncate_text(self, text, max_tokens):
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            truncated_text = encoding.decode(tokens[:max_tokens])
            return truncated_text
        return text

    def get_openai_embedding(self, text):
        text = text.replace("\n", " ")
        truncated_text = self.truncate_text(text, 8100) 
        try:
            response = self.client.embeddings.create(
                input=truncated_text,
                model="text-embedding-3-small"
            )
        except BadRequestError as e:
            print(f"Error: {e}, retrying with smaller input.")
            truncated_text = self.truncate_text(text, 7500
            )  
            response = self.client.embeddings.create(
                input=truncated_text,
                model="text-embedding-3-small"
            )
        return response.data[0].embedding

    def rerank(self, query, bm25_results):
        doc_ids = [i[-1].replace(".txt","").split("/")[-1] for i in bm25_results]
        doc_scores = [int(i[0]) for i in bm25_results]

        # doc_texts = [self.mapper[doc_id]['desc'] for doc_id in doc_ids]
        embedded_docs = np.array([self.doc_vectorizer[str(doc_id)] for doc_id in doc_ids])
        embedded_query = np.array(self.get_openai_embedding(query)).reshape(1, -1)

        dot_products = np.dot(embedded_docs, embedded_query.T)  # A·B
        doc_norms = np.linalg.norm(embedded_docs, axis=1, keepdims=True)  # ||A||
        query_norm = np.linalg.norm(embedded_query)  # ||B||

        cos_sims = (dot_products / (doc_norms * query_norm)).flatten()

        q_features = np.column_stack([
            doc_scores,  # BM25 scores
            cos_sims
        ])

        
        clf = xgb.XGBClassifier() 
        clf.load_model('/app/finsearch/retrieval/model/constant/model_openai.json')

        q_probs = clf.predict_proba(q_features)[:, 1]

        combined = list(zip(doc_ids, q_probs))
        combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

        ranked_doc_ids = [doc_id for doc_id, _ in combined_sorted]
        return ranked_doc_ids
    
class BM25RetrieverTFIDF():
    def __init__(self, base:BM25Retriever, config:BM25Config, mapper: dict):
        self.config = config
        self.base = base
        self.mapper = mapper
        self.tfidf_vectorizer = TfidfVectorizer()

    async def retrieve(self, query: str, k: int = 1) -> List[str]:
        bm25_results = self.base.bsbi_instance.retrieve_bm25_taat(query, k=k)
        bm25_results = self.rerank(query, bm25_results)
        return bm25_results


    def rerank(self, query, bm25_results):
        doc_ids = [i[-1].replace(".txt","").split("/")[-1] for i in bm25_results]
        doc_scores = [int(i[0]) for i in bm25_results]

        doc_texts = [self.mapper[doc_id]['desc'] for doc_id in doc_ids]

        self.tfidf_vectorizer.fit(doc_texts + [query])
        embedded_docs = self.tfidf_vectorizer.transform(doc_texts)
        embedded_query = self.tfidf_vectorizer.transform([query])

        embedded_docs_dense = embedded_docs.toarray()
        embedded_query_dense = embedded_query.toarray()

        dot_products = np.dot(embedded_docs_dense, embedded_query_dense.T)  # A·B
        doc_norms = np.linalg.norm(embedded_docs_dense, axis=1, keepdims=True)  # ||A||
        query_norm = np.linalg.norm(embedded_query_dense)  # ||B||

        cos_sims = (dot_products / (doc_norms * query_norm)).flatten()

        q_features = np.column_stack([
            doc_scores,  # BM25 scores
            cos_sims
        ])

        
        clf = xgb.XGBClassifier() 
        clf.load_model('/app/finsearch/retrieval/model/constant/model_tfidf.json')

        q_probs = clf.predict_proba(q_features)[:, 1]

        combined = list(zip(doc_ids, q_probs))
        combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

        ranked_doc_ids = [doc_id for doc_id, _ in combined_sorted]
        return ranked_doc_ids




if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir='arxiv_collections',
                              postings_encoding=VBEPostings,
                              output_dir='/app/index')
    BSBI_instance.do_indexing()  # memulai indexing!