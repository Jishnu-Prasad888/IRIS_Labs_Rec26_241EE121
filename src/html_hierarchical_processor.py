# src/html_hierarchical_processor.py
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path

nltk.download("punkt", quiet=True)

@dataclass
class DocumentNode:
    """Node in document hierarchy for HTML content"""
    id: str
    text: str
    level: int  # 0: document, 1: book/chapter, 2: section, 3: paragraph
    parent_id: Optional[str]
    children_ids: List[str]
    metadata: Dict
    tag_name: str


class HTMLHierarchicalProcessor:
    """
    Process HTML with logical/semantic segmentation for The Odyssey
    """
    
    def __init__(self, html_path: str):
        self.html_path = html_path
        self.nodes: Dict[str, DocumentNode] = {}
        self.root_nodes: List[str] = []
        
        # Odyssey-specific patterns
        self.book_patterns = [
            (r'^BOOK\s+([IVXLCDM]+)', 1),  # BOOK I, BOOK II, etc.
            (r'^Book\s+([IVXLCDM]+)', 1),
            (r'^BOOK\s+(\d+)', 1),
            (r'^CHAPTER\s+([IVXLCDM]+)', 1),
            (r'^Chapter\s+([IVXLCDM]+)', 1),
        ]
        
        # Section patterns for HTML structure
        self.section_patterns = [
            (r'^[IVXLCDM]+\.', 2),  # Roman numerals followed by dot
            (r'^\d+\.', 2),         # Numbers followed by dot
            (r'^[A-Z]\.', 3),       # Capital letters followed by dot
        ]
    
    def _identify_book_chapter(self, text: str) -> Tuple[Optional[str], int]:
        """Identify if text is a book/chapter header"""
        for pattern, level in self.book_patterns:
            match = re.match(pattern, text.strip())
            if match:
                return match.group(0), level
        return None, 0
    
    def _identify_section(self, text: str) -> Tuple[Optional[str], int]:
        """Identify if text is a section header"""
        for pattern, level in self.section_patterns:
            match = re.match(pattern, text.strip())
            if match:
                return match.group(0), level
        return None, 0
    
    def process_html(self) -> List[DocumentNode]:
        """Process Odyssey HTML with semantic structure"""
        with open(self.html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        
        # Remove unwanted tags
        for tag in soup(["style", "script", "meta", "link", "head"]):
            tag.decompose()
        
        content = soup.body or soup
        nodes = []
        node_id = 0
        
        # Track current hierarchy
        current_hierarchy: Dict[str, Optional[str]] = {
            "book": None,
            "section": None,
            "subsection": None
        }
        
        # Process all text elements
        for element in content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "div"]):
            if element.name in ["script", "style"]:
                continue
            
            text = element.get_text(" ", strip=True)
            if not text:
                continue
            
            # Check if this is a book/chapter header
            book_match, book_level = self._identify_book_chapter(text)
            if book_match:
                # Create book node
                book_node = DocumentNode(
                    id=f"book_{node_id}",
                    text=text,
                    level=1,
                    parent_id=None,
                    children_ids=[],
                    metadata={
                        "type": "book",
                        "book_id": book_match,
                        "tag": element.name
                    },
                    tag_name=element.name
                )
                nodes.append(book_node)
                self.nodes[book_node.id] = book_node
                self.root_nodes.append(book_node.id)
                
                # Update hierarchy
                current_hierarchy["book"] = book_node.id
                current_hierarchy["section"] = None
                current_hierarchy["subsection"] = None
                
                node_id += 1
                continue
            
            # Check if this is a section header
            section_match, section_level = self._identify_section(text)
            if section_match and len(text.split()) < 20:  # Likely a header, not paragraph
                section_node = DocumentNode(
                    id=f"section_{node_id}",
                    text=text,
                    level=2,
                    parent_id=current_hierarchy["book"],
                    children_ids=[],
                    metadata={
                        "type": "section",
                        "section_id": section_match,
                        "tag": element.name
                    },
                    tag_name=element.name
                )
                nodes.append(section_node)
                self.nodes[section_node.id] = section_node
                
                # Add to parent's children
                if section_node.parent_id and section_node.parent_id in self.nodes:
                    self.nodes[section_node.parent_id].children_ids.append(section_node.id)
                
                # Update hierarchy
                current_hierarchy["section"] = section_node.id
                current_hierarchy["subsection"] = None
                
                node_id += 1
                continue
            
            # Regular paragraph or content
            # Check if it's substantial content (not just a few words)
            if len(text.split()) > 10:
                # Determine parent: subsection > section > book
                parent_id = None
                if current_hierarchy["subsection"]:
                    parent_id = current_hierarchy["subsection"]
                elif current_hierarchy["section"]:
                    parent_id = current_hierarchy["section"]
                    # Check if we should create a subsection
                    if len(text.split()) < 50 and ":" in text:  # Likely a subsection header
                        # Create subsection
                        subsection_node = DocumentNode(
                            id=f"subsection_{node_id}",
                            text=text,
                            level=3,
                            parent_id=current_hierarchy["section"],
                            children_ids=[],
                            metadata={
                                "type": "subsection",
                                "tag": element.name
                            },
                            tag_name=element.name
                        )
                        nodes.append(subsection_node)
                        self.nodes[subsection_node.id] = subsection_node
                        
                        if subsection_node.parent_id and subsection_node.parent_id in self.nodes:
                            self.nodes[subsection_node.parent_id].children_ids.append(subsection_node.id)
                        
                        current_hierarchy["subsection"] = subsection_node.id
                        parent_id = subsection_node.id
                        node_id += 1
                        continue
                elif current_hierarchy["book"]:
                    parent_id = current_hierarchy["book"]
                else:
                    # No parent found, skip
                    continue
                
                # Create paragraph node
                para_node = DocumentNode(
                    id=f"para_{node_id}",
                    text=text,
                    level=4,
                    parent_id=parent_id,
                    children_ids=[],
                    metadata={
                        "type": "paragraph",
                        "word_count": len(text.split()),
                        "tag": element.name
                    },
                    tag_name=element.name
                )
                nodes.append(para_node)
                self.nodes[para_node.id] = para_node
                
                # Add to parent's children
                if para_node.parent_id and para_node.parent_id in self.nodes:
                    self.nodes[para_node.parent_id].children_ids.append(para_node.id)
                
                node_id += 1
        
        print(f"Created {len(nodes)} hierarchical nodes")
        print(f"Root nodes (books/chapters): {len(self.root_nodes)}")
        
        return nodes
    
    def create_semantic_chunks(self, min_words: int = 50, max_words: int = 300) -> List[Dict]:
        """Create chunks based on semantic structure"""
        chunks = []
        
        # Strategy 1: Use book-level chunks for overview
        for root_id in self.root_nodes:
            root_node = self.nodes[root_id]
            chunk_text = self._collect_subtree_text(root_id, max_depth=2, max_words=max_words)
            
            if chunk_text and len(chunk_text.split()) >= min_words:
                chunks.append({
                    "chunk_id": root_id,
                    "text": chunk_text,
                    "metadata": {
                        **root_node.metadata,
                        "chunk_type": "book_overview",
                        "level": root_node.level,
                        "parent_id": root_node.parent_id,
                        "children_count": len(root_node.children_ids)
                    }
                })
        
        # Strategy 2: Use section-level chunks for detailed content
        for node_id, node in self.nodes.items():
            if node.level == 2:  # Section level
                chunk_text = self._collect_subtree_text(node_id, max_depth=3, max_words=max_words)
                
                if chunk_text and len(chunk_text.split()) >= min_words:
                    chunks.append({
                        "chunk_id": node_id,
                        "text": chunk_text,
                        "metadata": {
                            **node.metadata,
                            "chunk_type": "section_detail",
                            "level": node.level,
                            "parent_id": node.parent_id,
                            "children_count": len(node.children_ids)
                        }
                    })
        
        # Strategy 3: Individual paragraphs for very specific content
        for node_id, node in self.nodes.items():
            if node.level == 4 and len(node.text.split()) >= 30:  # Substantial paragraphs
                chunks.append({
                    "chunk_id": node_id,
                    "text": node.text,
                    "metadata": {
                        **node.metadata,
                        "chunk_type": "paragraph",
                        "level": node.level,
                        "parent_id": node.parent_id,
                        "children_count": 0
                    }
                })
        
        # Remove duplicates and sort by hierarchy level
        unique_chunks = {}
        for chunk in chunks:
            if chunk["chunk_id"] not in unique_chunks:
                unique_chunks[chunk["chunk_id"]] = chunk
            else:
                # Keep the more comprehensive chunk
                if len(chunk["text"]) > len(unique_chunks[chunk["chunk_id"]]["text"]):
                    unique_chunks[chunk["chunk_id"]] = chunk
        
        final_chunks = list(unique_chunks.values())
        final_chunks.sort(key=lambda x: x["metadata"]["level"])
        
        print(f"Created {len(final_chunks)} semantic chunks")
        return final_chunks
    
    def _collect_subtree_text(self, node_id: str, max_depth: int = 2, max_words: int = 300) -> str:
        """Collect text from node and its children up to max_depth"""
        if node_id not in self.nodes:
            return ""
        
        node = self.nodes[node_id]
        texts = [node.text]
        current_words = len(node.text.split())
        
        if max_depth > 0 and node.children_ids:
            for child_id in node.children_ids[:10]:  # Limit children
                if current_words >= max_words:
                    break
                
                child_text = self._collect_subtree_text(child_id, max_depth - 1, max_words - current_words)
                if child_text:
                    texts.append(child_text)
                    current_words += len(child_text.split())
        
        return "\n\n".join(texts)
    
    def save_hierarchy(self, output_path: str):
        """Save hierarchy to JSON file"""
        hierarchy_data = {
            "nodes": {k: self._node_to_dict(v) for k, v in self.nodes.items()},
            "root_nodes": self.root_nodes
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(hierarchy_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved hierarchy to {output_path}")
    
    def _node_to_dict(self, node: DocumentNode) -> Dict:
        """Convert DocumentNode to serializable dict"""
        return {
            "id": node.id,
            "text_preview": node.text[:100] + "..." if len(node.text) > 100 else node.text,
            "text_length": len(node.text),
            "level": node.level,
            "parent_id": node.parent_id,
            "children_ids": node.children_ids,
            "metadata": node.metadata,
            "tag_name": node.tag_name
        }