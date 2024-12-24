# INFORMASI PENTING:
# Silakan merujuk pada slide di SCeLE tentang Query Auto-Completion (13-14)
# untuk referensi implementasi struktur data trie.
import heapq

class TrieNode:
    """
    Abstraksi node dalam suatu struktur data trie.
    """
    def __init__(self, char):
        self.char = char
        self.freq = 0
        self.children = {}

    def __str__(self):
        return self.char

class Trie:
    """
    Abstraksi struktur data trie.
    """
    def __init__(self):
        self.root = TrieNode("")

    def insert(self, word, freq):
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node
        node.freq += freq

    def __get_last_node(self, query):
        """
        Method ini mengambil node terakhir yang berasosiasi dengan suatu kata.
        Misalnya untuk query "halo", maka node terakhir adalah node "o"
        Jika no match, cukup return None saja.

        Parameters
        ----------
        query: str
            query yang ingin dilengkapi
        
        Returns
        -------
        TrieNode
            node terakhir dari suatu query, atau None
            jika tidak match
        """
        # TODO
        node = self.root
        for char in query:
            if char in node.children:
                node = node.children[char]
            else:
                return None
        return node

    def __get_all_next_subwords(self, query):
        """
        Method ini melakukan traversal secara DFS untuk mendapatkan semua
        subwords yang mengikuti suatu query yang diberikan beserta dengan 
        frekuensi kemunculannya dalam struktur data dictionary. Silakan membuat
        fungsi helper jika memang dibutuhkan.

        Jika tidak ada match, return dictionary kosong saja.

        Parameters
        ----------
        query: str
            query yang ingin dilengkapi
        
        Returns
        -------
        dict(str, int)
            dictionary dengan key berupa kandidat subwords dan value berupa
            frekuensi kemunculan subwords tersebut
        """
        # TODO
        node = self.__get_last_node(query)
        if node is None:
            return {}
        
        base_path = []
        subwords = {}
        def dfs(current_node, path):
            if current_node.freq > 0:
                subword = ''.join(path)
                subwords[subword] = current_node.freq
            for child_char, child_node in current_node.children.items():
                dfs(child_node, path + [child_char])
        

        dfs(node, base_path)
        return subwords

    def get_recommendations(self, query, k=5):
        """
        Method ini mengembalikan top-k rekomendasi subwords untuk melanjutkan
        query yang diberikan. Urutkan berdasarkan value (frekuensi) kemunculan
        subwords secara descending.

        Parameters
        ----------
        query: str
            query yang ingin dilengkapi
        k: int
            top-k subwords yang paling banyak frekuensinya
        
        Returns
        -------
        List[str]
            top-k subwords yang paling "matched"
        """
        # TODO
        next_subwords = self.__get_all_next_subwords(query)
        if len(next_subwords) == 0:
            return []
        

        sorted_next_subwords = sorted(next_subwords.items(), key=lambda x: (-x[1], x[0]))
        top_k_next_subwords = [word for word, _ in sorted_next_subwords[:k]]
        return top_k_next_subwords
    

class RMQTrieNode:
    """
    Abstraksi node dalam suatu struktur data trie dengan RMQ optimasi.
    """
    def __init__(self, char):
        self.char = char
        self.freq = 0
        self.children = {}
        self.max_freq = 0  # Menyimpan frekuensi maksimum di subtree

    def __str__(self):
        return self.char
        
class RMQTrie:
    """
    Abstraksi struktur data trie dengan RMQ optimasi.
    """
    def __init__(self):
        self.root = RMQTrieNode("")

    def insert(self, word, freq):
        node = self.root
        nodes_on_path = [node]
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                new_node = RMQTrieNode(char)
                node.children[char] = new_node
                node = new_node
            nodes_on_path.append(node)
        node.freq += freq
        # Update max_freq along the path
        for node in reversed(nodes_on_path):
            old_max_freq = node.max_freq
            node.max_freq = max(node.freq, max([child.max_freq for child in node.children.values()] or [0]))
            if node.max_freq == old_max_freq:
                break

    def __get_last_node(self, query):
        """
        Method ini mengambil node terakhir yang berasosiasi dengan suatu kata.
        Jika no match, return None.

        Parameters
        ----------
        query: str
            query yang ingin dilengkapi
        
        Returns
        -------
        RMQTrieNode
            node terakhir dari suatu query, atau None
            jika tidak match
        """
        node = self.root
        for char in query:
            if char in node.children:
                node = node.children[char]
            else:
                return None
        return node

    def get_recommendations(self, query, k=5):
        """
        Method ini mengembalikan top-k rekomendasi subwords untuk melanjutkan
        query yang diberikan. Urutkan berdasarkan frekuensi kemunculan
        subwords secara descending, dan lex order jika frekuensi sama.

        Parameters
        ----------
        query: str
            query yang ingin dilengkapi
        k: int
            top-k subwords yang paling banyak frekuensinya
        
        Returns
        -------
        List[str]
            top-k subwords yang paling "matched"
        """
        node = self.__get_last_node(query)
        if node is None:
            return []
        heap = []
        # Menggunakan heap sebagai max-heap (dengan frekuensi negatif)
        heapq.heappush(heap, (-node.max_freq, query, node))
        recommendations = []
        collected_words = set()
        while heap and len(recommendations) < k:
            neg_max_freq, word_so_far, curr_node = heapq.heappop(heap)
            if curr_node.freq > 0 and word_so_far not in collected_words:
                recommendations.append(word_so_far[len(query):])  # hanya suffix setelah query
                collected_words.add(word_so_far)
            for child_char, child_node in curr_node.children.items():
                new_word = word_so_far + child_char
                heapq.heappush(heap, (-child_node.max_freq, new_word, child_node))
        return recommendations
    
    
if __name__ == '__main__':
    # contoh dari slide
    trie = RMQTrie()
    trie.insert("nba", 5)
    trie.insert("news", 6)
    trie.insert("nab", 8)
    trie.insert("ngv", 9)
    trie.insert("netflix", 7)
    trie.insert("netbank", 11)
    trie.insert("network", 10)
    trie.insert("netball", 3)
    trie.insert("netbeans", 4)


    assert trie.get_recommendations('n') == ['etbank', 'etwork', 'gv', 'ab', 'etflix'], "output salah"
    assert trie.get_recommendations('') == ['netbank', 'network', 'ngv', 'nab', 'netflix'], "output salah"
    assert trie.get_recommendations('a') == [], "output salah"
    assert trie.get_recommendations('na') == ['b'], "output salah"