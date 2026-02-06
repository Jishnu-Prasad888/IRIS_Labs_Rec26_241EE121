# src/advanced_data_processor.py
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
import fitz  # PyMuPDF for PDF processing
from pathlib import Path

@dataclass
class DocumentNode:
    """Represents a node in the document hierarchy"""
    id: str
    text: str
    level: int  # 0: document, 1: chapter, 2: section, 3: subsection, 4: paragraph
    parent_id: Optional[str]
    children_ids: List[str]
    metadata: Dict
    start_page: Optional[int] = None
    end_page: Optional[int] = None


class AdvancedDataProcessor:
    """
    Implements logical/semantic segmentation and hierarchical structuring
    Inspired by TreeRAG and document segmentation research
    """
    
    def __init__(self, content_path: str, content_type: str = "html"):
        self.content_path = content_path
        self.content_type = content_type
        self.nodes: Dict[str, DocumentNode] = {}
        self.root_nodes: List[str] = []
        
        # Regex patterns for identifying logical sections
        self.section_patterns = [
            (r'^(CHAPTER|Chapter|CHAP\.)\s+([IVXLCDM0-9]+|[A-Z])', 1),
            (r'^(\d+\.\d+)\s+', 2),  # 1.1, 2.3, etc.
            (r'^(\d+\.\d+\.\d+)\s+', 3),  # 1.1.1, 2.3.4, etc.
            (r'^(\d+)\s+', 2),  # 1, 2, 3
            (r'^([A-Z])\.\s+', 3),  # A., B., C.
            (r'^\(([a-z])\)\s+', 4),  # (a), (b), (c)
        ]
        
    def _determine_level(self, text: str) -> Tuple[Optional[str], int]:
        """Identify if text is a section header and determine its level"""
        for pattern, level in self.section_patterns:
            match = re.match(pattern, text.strip())
            if match:
                return match.group(1), level
        return None, 0  # Regular paragraph
    
    def process_pdf(self) -> List[DocumentNode]:
        """Process PDF with logical segmentation"""
        doc = fitz.open(self.content_path)
        nodes = []
        current_hierarchy = [None] * 5  # Track current nodes at each level
        node_id = 0
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            # Split by potential section headers
            lines = text.split('\n')
            current_text = []
            
            for line in lines:
                section_id, level = self._determine_level(line)
                
                if section_id and level > 0:
                    # Save previous content if exists
                    if current_text:
                        # Create paragraph node
                        para_node = DocumentNode(
                            id=f"para_{node_id}",
                            text='\n'.join(current_text),
                            level=4,
                            parent_id=current_hierarchy[3] or current_hierarchy[2] or current_hierarchy[1],
                            children_ids=[],
                            metadata={
                                "page": page_num,
                                "type": "paragraph"
                            }
                        )
                        nodes.append(para_node)
                        node_id += 1
                        current_text = []
                    
                    # Create section node
                    section_node = DocumentNode(
                        id=f"sec_{node_id}",
                        text=line,
                        level=level,
                        parent_id=current_hierarchy[level-1] if level > 1 else None,
                        children_ids=[],
                        metadata={
                            "section_id": section_id,
                            "page": page_num,
                            "type": "section"
                        }
                    )
                    
                    # Update hierarchy
                    current_hierarchy[level] = section_node.id
                    # Clear lower levels
                    for l in range(level + 1, 5):
                        current_hierarchy[l] = None
                    
                    # Add to parent's children
                    if section_node.parent_id:
                        parent = next(n for n in nodes if n.id == section_node.parent_id)
                        parent.children_ids.append(section_node.id)
                    
                    nodes.append(section_node)
                    node_id += 1
                    
                    # If this is a root-level section
                    if level == 1 and section_node.parent_id is None:
                        self.root_nodes.append(section_node.id)
                        
                else:
                    # Regular text line
                    if line.strip():
                        current_text.append(line.strip())
            
            # Add remaining text from page
            if current_text:
                para_node = DocumentNode(
                    id=f"para_{node_id}",
                    text='\n'.join(current_text),
                    level=4,
                    parent_id=current_hierarchy[3] or current_hierarchy[2] or current_hierarchy[1],
                    children_ids=[],
                    metadata={
                        "page": page_num,
                        "type": "paragraph"
                    }
                )
                nodes.append(para_node)
                node_id += 1
        
        # Store nodes in dictionary for easy lookup
        self.nodes = {node.id: node for node in nodes}
        return nodes
    
    def process_html(self) -> List[DocumentNode]:
        """Process HTML with semantic tagging"""
        with open(self.content_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        
        nodes = []
        current_hierarchy = [None] * 5
        node_id = 0
        
        # Map HTML tags to hierarchy levels
        tag_to_level = {
            'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5,
            'section': 2, 'article': 1, 'div': 3
        }
        
        def process_element(el, current_hierarchy):
            nonlocal node_id
            
            if el.name in ['style', 'script', 'meta', 'link', 'head']:
                return current_hierarchy
            
            # Check if element is a structural tag
            if el.name in tag_to_level:
                level = tag_to_level[el.name]
                text = el.get_text(" ", strip=True)
                
                if text:
                    # Create node for this element
                    node = DocumentNode(
                        id=f"node_{node_id}",
                        text=text,
                        level=level,
                        parent_id=current_hierarchy[level-1] if level > 1 else None,
                        children_ids=[],
                        metadata={
                            "tag": el.name,
                            "type": "structural"
                        }
                    )
                    
                    # Update hierarchy
                    current_hierarchy[level] = node.id
                    # Clear lower levels
                    for l in range(level + 1, 5):
                        current_hierarchy[l] = None
                    
                    # Add to parent's children
                    if node.parent_id and node.parent_id in self.nodes:
                        self.nodes[node.parent_id].children_ids.append(node.id)
                    
                    nodes.append(node)
                    self.nodes[node.id] = node
                    node_id += 1
                    
                    if level == 1:
                        self.root_nodes.append(node.id)
            
            # Process children
            if hasattr(el, 'children'):
                for child in el.children:
                    if hasattr(child, 'name'):
                        current_hierarchy = process_element(child, current_hierarchy)
                    elif isinstance(child, str) and child.strip():
                        # Text node - create paragraph
                        node = DocumentNode(
                            id=f"para_{node_id}",
                            text=child.strip(),
                            level=4,
                            parent_id=current_hierarchy[3] or current_hierarchy[2] or current_hierarchy[1],
                            children_ids=[],
                            metadata={
                                "type": "paragraph",
                                "tag": "text"
                            }
                        )
                        nodes.append(node)
                        self.nodes[node.id] = node
                        node_id += 1
            
            return current_hierarchy
        
        process_element(soup.body or soup, current_hierarchy)
        return nodes
    
    def create_logical_chunks(self) -> List[Dict]:
        """Create chunks based on logical structure"""
        chunks = []
        
        # Traverse hierarchy and create chunks at appropriate levels
        for node_id in self.root_nodes:
            node = self.nodes[node_id]
            
            # For root-level sections, decide chunking strategy
            if node.level <= 2:  # Document, chapter, major section
                # Include children's text
                chunk_text = self._get_subtree_text(node_id)
                
                if chunk_text:
                    chunks.append({
                        "chunk_id": node_id,
                        "text": chunk_text,
                        "metadata": {
                            **node.metadata,
                            "level": node.level,
                            "type": "logical_chunk",
                            "node_type": node.metadata.get("type", "unknown"),
                            "parent_id": node.parent_id,
                            "children_count": len(node.children_ids)
                        }
                    })
            else:
                # Lower level nodes as individual chunks
                chunks.append({
                    "chunk_id": node_id,
                    "text": node.text,
                    "metadata": {
                        **node.metadata,
                        "level": node.level,
                        "type": "atomic_chunk",
                        "node_type": node.metadata.get("type", "unknown"),
                        "parent_id": node.parent_id
                    }
                })
        
        return chunks
    
    def _get_subtree_text(self, node_id: str, max_depth: int = 2) -> str:
        """Get text from a node and its children up to certain depth"""
        node = self.nodes[node_id]
        texts = [node.text]
        
        if max_depth > 0 and node.children_ids:
            for child_id in node.children_ids[:10]:  # Limit children
                child_text = self._get_subtree_text(child_id, max_depth - 1)
                if child_text:
                    texts.append(child_text)
        
        return "\n\n".join(texts)
    
    def save_hierarchy(self, output_path: str):
        """Save hierarchical structure to JSON"""
        hierarchy = {
            "nodes": {k: self._node_to_dict(v) for k, v in self.nodes.items()},
            "root_nodes": self.root_nodes
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)
    
    def _node_to_dict(self, node: DocumentNode) -> Dict:
        return {
            "id": node.id,
            "text": node.text[:500] + "..." if len(node.text) > 500 else node.text,
            "level": node.level,
            "parent_id": node.parent_id,
            "children_ids": node.children_ids,
            "metadata": node.metadata
        }