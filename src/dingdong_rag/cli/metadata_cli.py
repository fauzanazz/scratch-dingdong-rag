import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict, Counter

from ..metadata.enhanced_metadata import (
    InformatikaMetadataExtractor, 
    DocumentMetadata,
    enhance_document_metadata,
    generate_search_suggestions,
    create_search_filters
)


def extract_metadata_command(args):
    """Extract and display metadata for documents."""
    print(f"Extracting metadata from: {args.documents_dir}")
    
    extractor = InformatikaMetadataExtractor()
    documents_path = Path(args.documents_dir)
    
    if not documents_path.exists():
        print(f"❌ Directory not found: {args.documents_dir}")
        return 1
    
    # Find all PDF files
    if args.recursive:
        pdf_files = list(documents_path.glob("**/*.pdf"))
    else:
        pdf_files = list(documents_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ No PDF files found in {args.documents_dir}")
        return 1
    
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    if args.max_files and len(pdf_files) > args.max_files:
        pdf_files = pdf_files[:args.max_files]
        print(f"📝 Limited to {args.max_files} files for processing")
    
    # Extract metadata
    file_paths = [str(f) for f in pdf_files]
    metadata_results = extractor.extract_batch_metadata(file_paths)
    
    # Display results
    if args.output_format == 'table':
        display_metadata_table(metadata_results)
    elif args.output_format == 'json':
        display_metadata_json(metadata_results, args.output_file)
    elif args.output_format == 'summary':
        display_metadata_summary(metadata_results)
    
    # Save to file if requested
    if args.output_file and args.output_format != 'json':
        save_metadata_results(metadata_results, args.output_file)
        print(f"💾 Results saved to: {args.output_file}")
    
    return 0


def analyze_metadata_command(args):
    """Analyze metadata patterns and generate insights."""
    if not Path(args.metadata_file).exists():
        print(f"❌ Metadata file not found: {args.metadata_file}")
        return 1
    
    # Load metadata
    with open(args.metadata_file, 'r', encoding='utf-8') as f:
        metadata_data = json.load(f)
    
    # Convert to DocumentMetadata objects
    metadata_results = {}
    for file_path, meta_dict in metadata_data.items():
        metadata_results[file_path] = DocumentMetadata(**meta_dict)
    
    print("📊 METADATA ANALYSIS")
    print("=" * 60)
    
    analyze_course_distribution(metadata_results)
    analyze_document_types(metadata_results)
    analyze_year_distribution(metadata_results)
    analyze_topic_distribution(metadata_results)
    analyze_confidence_scores(metadata_results)
    
    # Generate search suggestions
    suggestions = generate_search_suggestions(metadata_results)
    print("\n🔍 SEARCH SUGGESTIONS")
    print("=" * 40)
    for category, items in suggestions.items():
        print(f"\n{category.title()}:")
        for item in items[:10]:  # Show top 10
            print(f"  • {item}")
    
    return 0


def search_metadata_command(args):
    """Search documents using metadata filters."""
    if not Path(args.metadata_file).exists():
        print(f"❌ Metadata file not found: {args.metadata_file}")
        return 1
    
    # Load metadata
    with open(args.metadata_file, 'r', encoding='utf-8') as f:
        metadata_data = json.load(f)
    
    # Convert to DocumentMetadata objects
    metadata_results = {}
    for file_path, meta_dict in metadata_data.items():
        metadata_results[file_path] = DocumentMetadata(**meta_dict)
    
    # Apply search filters
    filtered_results = apply_search_filters(metadata_results, args)
    
    print(f"🔍 SEARCH RESULTS ({len(filtered_results)} matches)")
    print("=" * 60)
    
    if not filtered_results:
        print("❌ No documents match the specified criteria")
        return 0
    
    # Display results
    for file_path, metadata in filtered_results.items():
        print(f"\n📄 {Path(file_path).name}")
        print(f"   Course: {metadata.course_code} - {metadata.course_name}")
        print(f"   Type: {metadata.document_type}")
        print(f"   Year: {metadata.academic_year or 'Unknown'}")
        print(f"   Topic: {metadata.topic_category or 'General'}")
        print(f"   Tags: {', '.join(metadata.tags[:5])}")
        print(f"   Path: {file_path}")
        if args.verbose:
            print(f"   Confidence: {metadata.confidence_score:.2f}")
    
    return 0


def display_metadata_table(metadata_results: Dict[str, DocumentMetadata]):
    """Display metadata in table format."""
    print("\n📋 METADATA TABLE")
    print("=" * 120)
    print(f"{'File':<30} {'Course':<15} {'Type':<12} {'Year':<6} {'Topic':<20} {'Conf':<6}")
    print("-" * 120)
    
    for file_path, metadata in metadata_results.items():
        filename = Path(file_path).name[:28]
        course = f"{metadata.course_code}"[:13]
        doc_type = metadata.document_type[:10]
        year = str(metadata.academic_year or "")[:4]
        topic = (metadata.topic_category or "")[:18]
        confidence = f"{metadata.confidence_score:.2f}"
        
        print(f"{filename:<30} {course:<15} {doc_type:<12} {year:<6} {topic:<20} {confidence:<6}")


def display_metadata_json(metadata_results: Dict[str, DocumentMetadata], output_file: str = None):
    """Display metadata in JSON format."""
    json_data = {}
    for file_path, metadata in metadata_results.items():
        json_data[file_path] = metadata.to_dict()
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"💾 JSON metadata saved to: {output_file}")
    else:
        print(json.dumps(json_data, indent=2, ensure_ascii=False))


def display_metadata_summary(metadata_results: Dict[str, DocumentMetadata]):
    """Display metadata summary statistics."""
    print("\n📈 METADATA SUMMARY")
    print("=" * 50)
    
    total_files = len(metadata_results)
    print(f"Total Files: {total_files}")
    
    # Course distribution
    courses = [m.course_code for m in metadata_results.values() if m.course_code != "Unknown"]
    course_counts = Counter(courses)
    print(f"\nTop Courses:")
    for course, count in course_counts.most_common(5):
        print(f"  {course}: {count} files")
    
    # Document type distribution  
    doc_types = [m.document_type for m in metadata_results.values() if m.document_type != "Unknown"]
    type_counts = Counter(doc_types)
    print(f"\nDocument Types:")
    for doc_type, count in type_counts.most_common():
        print(f"  {doc_type}: {count} files")
    
    # Average confidence
    confidences = [m.confidence_score for m in metadata_results.values()]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    print(f"\nAverage Confidence Score: {avg_confidence:.3f}")
    
    # High confidence files
    high_conf_files = [m for m in metadata_results.values() if m.confidence_score > 0.7]
    print(f"High Confidence Files (>0.7): {len(high_conf_files)} ({len(high_conf_files)/total_files*100:.1f}%)")


def analyze_course_distribution(metadata_results: Dict[str, DocumentMetadata]):
    """Analyze course distribution."""
    courses = defaultdict(int)
    for metadata in metadata_results.values():
        if metadata.course_code != "Unknown":
            courses[f"{metadata.course_code} - {metadata.course_name}"] += 1
    
    print("\n📚 COURSE DISTRIBUTION")
    print("-" * 40)
    for course, count in sorted(courses.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{course}: {count} files")


def analyze_document_types(metadata_results: Dict[str, DocumentMetadata]):
    """Analyze document type distribution."""
    types = defaultdict(int)
    for metadata in metadata_results.values():
        if metadata.document_type != "Unknown":
            types[metadata.document_type] += 1
    
    print("\n📄 DOCUMENT TYPE DISTRIBUTION")
    print("-" * 40)
    for doc_type, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"{doc_type}: {count} files")


def analyze_year_distribution(metadata_results: Dict[str, DocumentMetadata]):
    """Analyze academic year distribution."""
    years = defaultdict(int)
    for metadata in metadata_results.values():
        if metadata.academic_year:
            years[metadata.academic_year] += 1
    
    print("\n📅 ACADEMIC YEAR DISTRIBUTION")
    print("-" * 40)
    for year, count in sorted(years.items(), reverse=True):
        print(f"{year}: {count} files")


def analyze_topic_distribution(metadata_results: Dict[str, DocumentMetadata]):
    """Analyze topic distribution."""
    topics = defaultdict(int)
    for metadata in metadata_results.values():
        if metadata.topic_category:
            topics[metadata.topic_category] += 1
    
    print("\n🏷️ TOPIC DISTRIBUTION")
    print("-" * 40)
    for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{topic}: {count} files")


def analyze_confidence_scores(metadata_results: Dict[str, DocumentMetadata]):
    """Analyze confidence score distribution."""
    scores = [m.confidence_score for m in metadata_results.values()]
    
    print("\n🎯 CONFIDENCE SCORE ANALYSIS")
    print("-" * 40)
    print(f"Average: {sum(scores)/len(scores):.3f}")
    print(f"Min: {min(scores):.3f}")
    print(f"Max: {max(scores):.3f}")
    
    # Score ranges
    ranges = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-1.0": 0}
    for score in scores:
        if score <= 0.3:
            ranges["0.0-0.3"] += 1
        elif score <= 0.5:
            ranges["0.3-0.5"] += 1
        elif score <= 0.7:
            ranges["0.5-0.7"] += 1
        else:
            ranges["0.7-1.0"] += 1
    
    print("Score Ranges:")
    for range_name, count in ranges.items():
        percentage = count/len(scores)*100
        print(f"  {range_name}: {count} files ({percentage:.1f}%)")


def apply_search_filters(metadata_results: Dict[str, DocumentMetadata], args) -> Dict[str, DocumentMetadata]:
    """Apply search filters to metadata results."""
    filtered = {}
    
    for file_path, metadata in metadata_results.items():
        # Apply course filter
        if args.course and args.course.lower() not in metadata.course_code.lower():
            continue
        
        # Apply type filter
        if args.doc_type and args.doc_type.lower() not in metadata.document_type.lower():
            continue
        
        # Apply year filter
        if args.year and str(args.year) != str(metadata.academic_year):
            continue
        
        # Apply topic filter
        if args.topic and metadata.topic_category:
            if args.topic.lower() not in metadata.topic_category.lower():
                continue
        
        # Apply semester filter
        if args.semester and args.semester.lower() not in metadata.semester.lower():
            continue
        
        # Apply solution filter
        if args.solutions_only and not metadata.is_solution:
            continue
        
        # Apply practice filter
        if args.practice_only and not metadata.is_practice:
            continue
        
        # Apply minimum confidence filter
        if metadata.confidence_score < args.min_confidence:
            continue
        
        filtered[file_path] = metadata
    
    return filtered


def save_metadata_results(metadata_results: Dict[str, DocumentMetadata], output_file: str):
    """Save metadata results to file."""
    json_data = {}
    for file_path, metadata in metadata_results.items():
        json_data[file_path] = metadata.to_dict()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced Metadata CLI for DingDong RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract metadata from documents')
    extract_parser.add_argument('documents_dir', help='Directory containing documents')
    extract_parser.add_argument('--recursive', '-r', action='store_true', default=True, help='Search recursively')
    extract_parser.add_argument('--max-files', type=int, help='Maximum number of files to process')
    extract_parser.add_argument('--output-format', choices=['table', 'json', 'summary'], default='table', help='Output format')
    extract_parser.add_argument('--output-file', '-o', help='Save results to file')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze metadata patterns')
    analyze_parser.add_argument('metadata_file', help='JSON file containing metadata')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search documents using metadata')
    search_parser.add_argument('metadata_file', help='JSON file containing metadata')
    search_parser.add_argument('--course', help='Filter by course code')
    search_parser.add_argument('--doc-type', help='Filter by document type')
    search_parser.add_argument('--year', type=int, help='Filter by academic year')
    search_parser.add_argument('--topic', help='Filter by topic category')
    search_parser.add_argument('--semester', help='Filter by semester')
    search_parser.add_argument('--solutions-only', action='store_true', help='Show only solution documents')
    search_parser.add_argument('--practice-only', action='store_true', help='Show only practice documents')
    search_parser.add_argument('--min-confidence', type=float, default=0.0, help='Minimum confidence score')
    search_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    try:
        if args.command == 'extract':
            return extract_metadata_command(args)
        elif args.command == 'analyze':
            return analyze_metadata_command(args)
        elif args.command == 'search':
            return search_metadata_command(args)
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())