import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DocumentMetadata:
    """Rich metadata for academic documents."""
    # File information
    file_path: str
    file_name: str
    file_extension: str
    file_size_bytes: int
    
    # Academic hierarchy
    program: str  # e.g., "Informatika"
    semester: str  # e.g., "Semester 2"
    course_code: str  # e.g., "IF1220"
    course_name: str  # e.g., "Matematika Diskrit"
    
    # Document classification
    document_type: str  # Slide, Referensi, Soal, Catatan
    academic_year: Optional[str]  # 2024, 2023, etc.
    topic_category: Optional[str]  # Extracted from filename
    part_number: Optional[int]  # For multi-part documents
    
    # Content hints
    language: str  # Indonesian, English
    document_format: str  # PDF, DOC, etc.
    is_solution: bool  # For solution manuals
    is_practice: bool  # For practice problems
    
    # Searchable tags
    tags: List[str]
    keywords: List[str]
    
    # Processing metadata
    extraction_method: str
    processing_timestamp: str
    confidence_score: float  # How confident we are in the metadata extraction
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for vector store."""
        return asdict(self)


class InformatikaMetadataExtractor:
    """Metadata extractor specialized for ITB Informatika document structure."""
    
    def __init__(self):
        # Course code mappings
        self.course_mappings = {
            'IF1210': 'Algoritma dan Pemrograman 1',
            'IF1220': 'Matematika Diskrit',
            'IF1221': 'Logika Komputasional', 
            'IF1230': 'Organisasi dan Arsitektur Komputer',
            'WI2002': 'Literasi Data dan Intelegensi Artifisial',
            'IF2110': 'Algoritma dan Pemrograman 2',
            'IF2120': 'Probabilitas dan Statistika',
            'IF2123': 'Aljabar Linier dan Geometri',
            'IF2130': 'Sistem Operasi',
            'IF2150': 'Rekayasa Perangkat Lunak',
            'IF2180': 'Socio Informatika dan Profesionalisme'
        }
        
        # Document type patterns
        self.type_patterns = {
            'slide': r'slide|materi|presentasi',
            'referensi': r'referensi|reference|textbook|handbook|diktat',
            'soal': r'soal|kuis|uts|uas|ujian|exam|quiz|latihan|tugas',
            'catatan': r'catatan|notes|rangkuman|summary'
        }
        
        # Topic extraction patterns
        self.topic_patterns = {
            # Math topics
            'aljabar_boolean': r'aljabar.?boolean|boolean.?algebra',
            'graf': r'graf|graph',
            'himpunan': r'himpunan|set|sets',
            'kombinatorial': r'kombinatorial|combinatorial|kombinatorika',
            'kompleksitas': r'kompleksitas|complexity',
            'induksi': r'induksi|induction',
            'rekursi': r'rekursi|recursion|rekurens',
            'relasi_fungsi': r'relasi.{0,10}fungsi|relation.{0,10}function',
            'teori_bilangan': r'teori.?bilangan|number.?theory',
            'pohon': r'pohon|tree',
            'logika': r'logika|logic',
            
            # Programming topics
            'algoritma': r'algoritma|algorithm',
            'pemrograman': r'pemrograman|programming',
            'struktur_data': r'struktur.?data|data.?structure',
            
            # Systems topics
            'sistem_operasi': r'sistem.?operasi|operating.?system',
            'arsitektur': r'arsitektur|architecture',
            'organisasi': r'organisasi|organization',
            
            # Other CS topics
            'probabilitas': r'probabilitas|probability',
            'statistika': r'statistika|statistics',
            'aljabar_linear': r'aljabar.?linear|linear.?algebra',
            'rekayasa': r'rekayasa|software.?engineering'
        }
        
        # Academic year patterns
        self.year_pattern = r'(20\d{2})'
        self.part_pattern = r'(?:bagian|part|bag)[\s\-]*(\d+)|(\d+)[\s\-]*(?:bagian|part|bag)'
        
    def extract_metadata(self, file_path: str, content_hint: str = "") -> DocumentMetadata:
        """Extract comprehensive metadata from file path and content."""
        path = Path(file_path)
        parts = path.parts
        
        # Initialize with defaults
        metadata = DocumentMetadata(
            file_path=str(path),
            file_name=path.name,
            file_extension=path.suffix.lower(),
            file_size_bytes=path.stat().st_size if path.exists() else 0,
            program="Unknown",
            semester="Unknown", 
            course_code="Unknown",
            course_name="Unknown",
            document_type="Unknown",
            academic_year=None,
            topic_category=None,
            part_number=None,
            language="Indonesian",
            document_format=path.suffix.upper().lstrip('.'),
            is_solution=False,
            is_practice=False,
            tags=[],
            keywords=[],
            extraction_method="enhanced_hierarchical",
            processing_timestamp=datetime.now().isoformat(),
            confidence_score=0.0
        )
        
        confidence_factors = []
        
        # Extract from path hierarchy
        try:
            # Find program (e.g., "Informatika")
            for part in parts:
                if "Informatika" in part:
                    metadata.program = "Informatika"
                    confidence_factors.append(0.2)
                    break
            
            # Find semester (e.g., "Semester 2")
            for part in parts:
                if "Semester" in part:
                    metadata.semester = part
                    confidence_factors.append(0.15)
                    break
            
            # Extract course code and name
            course_pattern = r'(IF\d{4}|WI\d{4})\s*-\s*(.+)'
            for part in parts:
                match = re.search(course_pattern, part)
                if match:
                    metadata.course_code = match.group(1)
                    metadata.course_name = match.group(2).strip()
                    confidence_factors.append(0.25)
                    break
            
            # Validate course mapping
            if metadata.course_code in self.course_mappings:
                expected_name = self.course_mappings[metadata.course_code]
                if expected_name in metadata.course_name:
                    confidence_factors.append(0.1)
            
            # Extract document type from path
            for part in parts:
                part_lower = part.lower()
                for doc_type, pattern in self.type_patterns.items():
                    if re.search(pattern, part_lower, re.IGNORECASE):
                        metadata.document_type = doc_type.title()
                        confidence_factors.append(0.15)
                        break
                if metadata.document_type != "Unknown":
                    break
                    
        except Exception as e:
            confidence_factors.append(-0.1)  # Penalty for parsing errors
        
        # Extract from filename
        filename_lower = metadata.file_name.lower()
        
        # Academic year extraction
        year_match = re.search(self.year_pattern, metadata.file_name)
        if year_match:
            metadata.academic_year = year_match.group(1)
            confidence_factors.append(0.1)
        
        # Part number extraction  
        part_match = re.search(self.part_pattern, filename_lower)
        if part_match:
            metadata.part_number = int(part_match.group(1) or part_match.group(2))
            confidence_factors.append(0.05)
        
        # Topic category extraction
        for topic, pattern in self.topic_patterns.items():
            if re.search(pattern, filename_lower, re.IGNORECASE):
                metadata.topic_category = topic.replace('_', ' ').title()
                metadata.keywords.append(topic.replace('_', ' '))
                confidence_factors.append(0.1)
                break
        
        # Solution detection
        if re.search(r'solusi|solution|jawaban|pembahasan', filename_lower):
            metadata.is_solution = True
            metadata.tags.append('solution')
            confidence_factors.append(0.05)
        
        # Practice/exercise detection
        path_lower = str(path).lower()
        if re.search(r'latihan|practice|exercise|soal|kuis|quiz|uas|uts|ujian', path_lower):
            metadata.is_practice = True
            metadata.tags.append('practice')
            confidence_factors.append(0.05)
        
        # Language detection (simple heuristic)
        indonesian_indicators = ['dan', 'atau', 'dengan', 'untuk', 'pada', 'dalam', 'yang', 'adalah']
        english_indicators = ['and', 'or', 'with', 'for', 'on', 'in', 'the', 'is', 'a', 'an']
        
        if content_hint:
            content_lower = content_hint.lower()
            indo_count = sum(1 for word in indonesian_indicators if word in content_lower)
            eng_count = sum(1 for word in english_indicators if word in content_lower)
            
            if eng_count > indo_count * 1.5:
                metadata.language = "English"
                confidence_factors.append(0.05)
            elif indo_count > 0:
                confidence_factors.append(0.05)
        
        # Generate additional tags
        metadata.tags.extend([
            metadata.course_code.lower(),
            metadata.semester.lower().replace(' ', '_'),
            metadata.document_type.lower()
        ])
        
        if metadata.academic_year:
            metadata.tags.append(f"year_{metadata.academic_year}")
        
        if metadata.topic_category:
            metadata.tags.append(metadata.topic_category.lower().replace(' ', '_'))
        
        # Calculate final confidence score
        metadata.confidence_score = max(0.0, min(1.0, sum(confidence_factors)))
        
        return metadata
    
    def extract_batch_metadata(self, file_paths: List[str], 
                             content_hints: Optional[Dict[str, str]] = None) -> Dict[str, DocumentMetadata]:
        """Extract metadata for multiple files."""
        results = {}
        content_hints = content_hints or {}
        
        for file_path in file_paths:
            hint = content_hints.get(file_path, "")
            try:
                metadata = self.extract_metadata(file_path, hint)
                results[file_path] = metadata
            except Exception as e:
                # Create minimal metadata for failed extractions
                results[file_path] = DocumentMetadata(
                    file_path=file_path,
                    file_name=Path(file_path).name,
                    file_extension=Path(file_path).suffix.lower(),
                    file_size_bytes=0,
                    program="Unknown",
                    semester="Unknown", 
                    course_code="Unknown",
                    course_name="Unknown",
                    document_type="Unknown",
                    academic_year=None,
                    topic_category=None,
                    part_number=None,
                    language="Unknown",
                    document_format=Path(file_path).suffix.upper().lstrip('.'),
                    is_solution=False,
                    is_practice=False,
                    tags=[],
                    keywords=[],
                    extraction_method="enhanced_hierarchical_failed",
                    processing_timestamp=datetime.now().isoformat(),
                    confidence_score=0.0
                )
        
        return results


def enhance_document_metadata(documents_results: Dict[str, Any]) -> Dict[str, DocumentMetadata]:
    """Enhance existing document results with rich metadata."""
    extractor = InformatikaMetadataExtractor()
    enhanced_metadata = {}
    
    content_hints = {}
    for file_path, doc_data in documents_results.items():
        if isinstance(doc_data, dict) and 'content' in doc_data:
            content_hints[file_path] = doc_data['content'][:1000]  # First 1000 chars as hint
        elif isinstance(doc_data, str):
            content_hints[file_path] = doc_data[:1000]
    
    file_paths = list(documents_results.keys())
    metadata_results = extractor.extract_batch_metadata(file_paths, content_hints)
    
    return metadata_results


def create_search_filters(metadata: DocumentMetadata) -> Dict[str, Any]:
    """Create search filter dictionary from metadata."""
    filters = {
        'program': metadata.program,
        'semester': metadata.semester, 
        'course_code': metadata.course_code,
        'document_type': metadata.document_type,
        'language': metadata.language
    }
    
    if metadata.academic_year:
        filters['academic_year'] = metadata.academic_year
    
    if metadata.topic_category:
        filters['topic_category'] = metadata.topic_category
    
    if metadata.part_number:
        filters['part_number'] = metadata.part_number
    
    if metadata.is_solution:
        filters['is_solution'] = True
    
    if metadata.is_practice:
        filters['is_practice'] = True
    
    return filters


def generate_search_suggestions(metadata_results: Dict[str, DocumentMetadata]) -> Dict[str, List[str]]:
    """Generate search suggestions based on available metadata."""
    suggestions = {
        'courses': set(),
        'topics': set(),
        'years': set(),
        'document_types': set(),
        'semesters': set()
    }
    
    for metadata in metadata_results.values():
        if metadata.course_code != "Unknown":
            suggestions['courses'].add(f"{metadata.course_code} - {metadata.course_name}")
        
        if metadata.topic_category:
            suggestions['topics'].add(metadata.topic_category)
        
        if metadata.academic_year:
            suggestions['years'].add(metadata.academic_year)
        
        if metadata.document_type != "Unknown":
            suggestions['document_types'].add(metadata.document_type)
        
        if metadata.semester != "Unknown":
            suggestions['semesters'].add(metadata.semester)
    
    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in suggestions.items()}