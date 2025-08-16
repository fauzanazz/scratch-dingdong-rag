import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass


@dataclass
class DomainMatch:
    """Result from domain detection."""
    domain: str
    confidence: float
    matched_keywords: List[str]
    course_code: Optional[str] = None


class DomainDetector:
    """Detects academic domain from query keywords."""
    
    def __init__(self):
        self.domain_keywords = {
            # Sistem Operasi (Operating Systems) - IF2130
            "sistem_operasi": {
                "course_code": "IF2130",
                "path_filter": "IF2130 - Sistem Operasi",
                "keywords": [
                    "sistem operasi", "proses", "thread", "penjadwalan", "scheduling",
                    "memori", "memory management", "virtual memory", "paging",
                    "file system", "sistem file", "directory", "direktori",
                    "protection", "proteksi", "domain", "access control",
                    "deadlock", "sinkronisasi", "synchronization", "semaphore",
                    "mutex", "critical section", "bagian kritis",
                    "kernel", "sistem call", "system call", "interrupt",
                    "cpu scheduling", "algoritma penjadwalan",
                    "operating system", "os", "process", "thread", "scheduler",
                    "memory", "virtual", "file", "protection", "security",
                    "concurrency", "parallel", "multithreading",
                ]
            },
            
            "matematika_diskrit": {
                "course_code": "IF1220",
                "path_filter": "IF1220 - Matematika Diskrit",
                "keywords": [
                    "matematika diskrit", "graf", "graph", "teori graf",
                    "logika", "proposisi", "predicate", "predikat",
                    "himpunan", "set", "kombinatorika", "kombinasi",
                    "permutasi", "relasi", "fungsi", "function",
                    "modular", "algoritma", "kompleksitas",
                    "rekursi", "induksi", "proof", "pembuktian",
                    "boolean", "aljabar boolean", "matriks",
                        "discrete mathematics", "discrete math", "graph theory",
                    "logic", "proposition", "set theory", "combinatorics",
                    "permutation", "combination", "relation", "recursion",
                    "induction", "boolean algebra", "matrix"
                ]
            },
            
            # Basis Data (Database) - IF2150
            "basis_data": {
                "course_code": "IF2150", 
                "path_filter": "IF2150 - Basis Data",
                "keywords": [
                    # Indonesian terms
                    "basis data", "database", "sql", "query", "tabel",
                    "relasi", "normalisasi", "indeks", "transaksi",
                    "dbms", "sistem manajemen basis data",
                    "entity relationship", "er diagram", "primary key",
                    "foreign key", "join", "select", "insert", "update",
                    "normal form", "1nf", "2nf", "3nf", "bcnf",
                    # English terms
                    "database", "sql", "table", "relation", "normalization",
                    "transaction", "acid", "schema", "query optimization"
                ]
            },
            
            # Algoritma dan Struktur Data - IF2211
            "algoritma_struktur_data": {
                "course_code": "IF2211",
                "path_filter": "IF2211 - Algoritma",
                "keywords": [
                    # Indonesian terms
                    "algoritma", "struktur data", "array", "linked list",
                    "stack", "queue", "tree", "pohon", "binary tree",
                    "heap", "hash", "sorting", "searching", "pencarian",
                    "pengurutan", "kompleksitas", "big o", "rekursi",
                    "divide and conquer", "dynamic programming",
                    "greedy", "backtracking", "graph algorithm",
                    "huffman", "kompresi", "compression", "encoding",
                    "dijkstra", "kruskal", "prim", "floyd warshall",
                    # English terms
                    "algorithm", "data structure", "complexity", "recursion",
                    "sorting", "searching", "tree", "graph", "optimization"
                ]
            },
            
            # Jaringan Komputer (Computer Networks) - IF2230  
            "jaringan_komputer": {
                "course_code": "IF2230",
                "path_filter": "IF2230 - Jaringan",
                "keywords": [
                    # Indonesian terms
                    "jaringan komputer", "jaringan", "protokol", "tcp/ip",
                    "routing", "switching", "osi", "model osi", "ethernet",
                    "wi-fi", "wireless", "network", "internet", "http",
                    "dns", "dhcp", "firewall", "security", "keamanan",
                    # English terms  
                    "computer network", "network", "protocol", "routing",
                    "switching", "osi model", "tcp", "ip", "ethernet",
                    "wireless", "security", "firewall"
                ]
            }
        }
        
        # Compile regex patterns for better matching
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        self.compiled_patterns = {}
        for domain, config in self.domain_keywords.items():
            patterns = []
            for keyword in config["keywords"]:
                # Create case-insensitive regex pattern
                # For multi-word terms, we need more flexible matching
                escaped_keyword = re.escape(keyword.lower())
                
                # For multi-word phrases, allow flexible spacing
                if ' ' in keyword:
                    # Replace spaces with flexible whitespace pattern
                    flexible_keyword = escaped_keyword.replace(r'\ ', r'\s+')
                    patterns.append(f"\\b{flexible_keyword}\\b")
                else:
                    # Single word - use word boundaries
                    patterns.append(f"\\b{escaped_keyword}\\b")
            
            # Combine all patterns for this domain
            combined_pattern = "|".join(patterns)
            self.compiled_patterns[domain] = re.compile(combined_pattern, re.IGNORECASE)
    
    def detect_domain(self, query: str, confidence_threshold: float = 0.3) -> Optional[DomainMatch]:
        """
        Detect the most likely academic domain from a query.
        
        Args:
            query: The search query text
            confidence_threshold: Minimum confidence score to return a match
            
        Returns:
            DomainMatch object if domain detected with sufficient confidence, None otherwise
        """
        if not query or not query.strip():
            return None
        
        query_lower = query.lower()
        domain_scores = {}
        domain_matches = {}
        
        # Score each domain based on keyword matches
        for domain, config in self.domain_keywords.items():
            matched_keywords = []
            pattern = self.compiled_patterns[domain]
            
            # Find all matches using search instead of findall to get actual matched text
            for keyword in config["keywords"]:
                keyword_pattern = re.compile(re.escape(keyword.lower()).replace(r'\ ', r'\s+'), re.IGNORECASE)
                if keyword_pattern.search(query_lower):
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                # Calculate confidence score
                # Base score: number of unique matches / total keywords in domain  
                unique_matches = set(matched_keywords)
                base_score = len(unique_matches) / len(config["keywords"])
                
                # Bonus for multiple matches
                match_bonus = min(0.3, len(matched_keywords) * 0.1)  
                
                # Bonus for exact course mentions
                course_bonus = 0.0
                if config["course_code"].lower() in query_lower:
                    course_bonus = 0.4
                
                # Total confidence
                confidence = min(1.0, base_score + match_bonus + course_bonus)
                
                domain_scores[domain] = confidence
                domain_matches[domain] = matched_keywords
        
        # Return the highest scoring domain if it meets threshold
        if domain_scores:
            best_domain = max(domain_scores.keys(), key=lambda domain: domain_scores[domain])
            best_score = domain_scores[best_domain]
            
            if best_score >= confidence_threshold:
                config = self.domain_keywords[best_domain]
                return DomainMatch(
                    domain=best_domain,
                    confidence=best_score,
                    matched_keywords=domain_matches[best_domain],
                    course_code=config["course_code"]
                )
        
        return None
    
    def get_domain_filter(self, query: str, confidence_threshold: float = 0.3) -> Optional[str]:
        """
        Get domain filter string for vector search based on query.
        
        Args:
            query: The search query
            confidence_threshold: Minimum confidence to apply filtering
            
        Returns:
            Path filter string for metadata filtering, or None if no domain detected
        """
        domain_match = self.detect_domain(query, confidence_threshold)
        if domain_match:
            config = self.domain_keywords[domain_match.domain]
            return config["path_filter"]
        return None
    
    def explain_detection(self, query: str) -> Dict:
        """
        Explain domain detection results for debugging.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary with detection details for all domains
        """
        explanation = {
            "query": query,
            "domains": {},
            "best_match": None
        }
        
        query_lower = query.lower()
        
        # Analyze all domains
        for domain, config in self.domain_keywords.items():
            matched_keywords = []
            
            # Find matches using the same logic as detect_domain
            for keyword in config["keywords"]:
                keyword_pattern = re.compile(re.escape(keyword.lower()).replace(r'\ ', r'\s+'), re.IGNORECASE)
                if keyword_pattern.search(query_lower):
                    matched_keywords.append(keyword)
            
            if matched_keywords:
                unique_matches = set(matched_keywords)
                base_score = len(unique_matches) / len(config["keywords"])
                match_bonus = min(0.3, len(matched_keywords) * 0.1)
                course_bonus = 0.4 if config["course_code"].lower() in query_lower else 0.0
                confidence = min(1.0, base_score + match_bonus + course_bonus)
                
                explanation["domains"][domain] = {
                    "matches": matched_keywords,
                    "unique_matches": list(unique_matches),
                    "base_score": base_score,
                    "match_bonus": match_bonus,
                    "course_bonus": course_bonus,
                    "confidence": confidence,
                    "course_code": config["course_code"],
                    "path_filter": config["path_filter"]
                }
        
        # Find best match
        if explanation["domains"]:
            best_domain = max(
                explanation["domains"].keys(),
                key=lambda domain: explanation["domains"][domain]["confidence"],
            )
            explanation["best_match"] = {
                "domain": best_domain,
                **explanation["domains"][best_domain]
            }
        
        return explanation


def create_domain_detector() -> DomainDetector:
    """Factory function to create domain detector."""
    return DomainDetector()


# Example usage and testing
if __name__ == "__main__":
    detector = create_domain_detector()
    
    # Test queries
    test_queries = [
        "apa itu domains of protection dalam sistem operasi",
        "kompresi huffman dalam algoritma",
        "teori graf matematika diskrit",
        "normalisasi basis data",
        "tcp/ip protocol jaringan komputer",
        "hello world programming",  # Should not match any domain
        "database normalization 3nf",
        "encoding huffman compression algorithm"
    ]
    
    print("Domain Detection Test Results:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        domain_match = detector.detect_domain(query)
        if domain_match:
            print(f"  Domain: {domain_match.domain}")
            print(f"  Course: {domain_match.course_code}")
            print(f"  Confidence: {domain_match.confidence:.2f}")
            print(f"  Keywords: {domain_match.matched_keywords}")
            
            filter_str = detector.get_domain_filter(query)
            print(f"  Filter: {filter_str}")
        else:
            print("  No domain detected")