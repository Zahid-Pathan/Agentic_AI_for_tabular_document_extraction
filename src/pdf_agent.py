"""
PDF Agent - Specialized agent for PDF table extraction using pdfplumber
"""
import logging
from typing import Dict, List, Any, Optional
import pdfplumber
from datetime import datetime

logger = logging.getLogger(__name__)

class PDFAgent:
    """Specialized agent for PDF table extraction using pdfplumber"""
    
    def __init__(self):
        self.agent_name = "PDF_Agent"
        logger.info(f"🔧 {self.agent_name}: Initialized")
    
    def extract_tables(self, pdf_path: str) -> Dict[str, Any]:
        """Extract tables from PDF using pdfplumber"""
        logger.info(f"📄 {self.agent_name}: Starting extraction from {pdf_path}")
        
        try:
            extraction_result = {
                "agent": self.agent_name,
                "extraction_method": "pdfplumber",
                "timestamp": datetime.now().isoformat(),
                "pdf_path": pdf_path,
                "tables_found": 0,
                "overall_confidence": 0.0,
                "tables": [],
                "raw_text_length": 0,
                "pages_processed": 0,
                "extraction_details": {}
            }
            
            with pdfplumber.open(pdf_path) as pdf:
                raw_text = ""
                all_tables = []
                
                for page_num, page in enumerate(pdf.pages):
                    logger.info(f"📄 {self.agent_name}: Processing page {page_num + 1}")
                    
                    # Extract text
                    page_text = page.extract_text() or ""
                    raw_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    
                    logger.info(f"📄 {self.agent_name}: Found {len(page_tables or [])} tables on page {page_num + 1}")
                    
                    for table_idx, table in enumerate(page_tables or []):
                        if table and len(table) > 0:
                            processed_table = self._process_table(
                                table, page_num + 1, table_idx
                            )
                            if processed_table:
                                all_tables.append(processed_table)
                
                # Update results
                extraction_result.update({
                    "tables_found": len(all_tables),
                    "tables": all_tables,
                    "raw_text_length": len(raw_text),
                    "pages_processed": len(pdf.pages),
                    "overall_confidence": self._calculate_overall_confidence(all_tables, raw_text),
                    "extraction_details": {
                        "total_pages": len(pdf.pages),
                        "text_extraction_success": len(raw_text) > 50,
                        "table_detection_method": "pdfplumber_native"
                    }
                })
            
            # Log results
            logger.info(f"📊 {self.agent_name}: Extraction completed")
            logger.info(f"   Tables found: {extraction_result['tables_found']}")
            logger.info(f"   Text length: {extraction_result['raw_text_length']} chars")
            logger.info(f"   Overall confidence: {extraction_result['overall_confidence']:.2f}")
            
            return extraction_result
            
        except Exception as e:
            error_msg = f"PDF extraction failed: {str(e)}"
            logger.error(f"❌ {self.agent_name}: {error_msg}")
            
            return {
                "agent": self.agent_name,
                "extraction_method": "pdfplumber",
                "error": error_msg,
                "tables_found": 0,
                "overall_confidence": 0.0,
                "tables": []
            }
    
    def _process_table(self, raw_table: List[List[str]], page_num: int, table_idx: int) -> Optional[Dict]:
        """Process a single table from pdfplumber"""
        try:
            if not raw_table or len(raw_table) < 1:
                return None
            
            # Clean the table data
            cleaned_table = []
            for row in raw_table:
                cleaned_row = []
                for cell in row:
                    cleaned_cell = str(cell).strip() if cell is not None else ""
                    cleaned_row.append(cleaned_cell)
                cleaned_table.append(cleaned_row)
            
            # Calculate basic metrics
            total_cells = sum(len(row) for row in cleaned_table)
            non_empty_cells = sum(1 for row in cleaned_table for cell in row if cell)
            
            # Determine table structure
            num_rows = len(cleaned_table)
            num_cols = max(len(row) for row in cleaned_table) if cleaned_table else 0

            # 🔹 NEW: build per-cell records with row/col metadata & inferred headers
            cell_records, header_info = self._build_cell_records(cleaned_table)

            table_info = {
                "table_id": f"pdf_table_{page_num}_{table_idx}",
                "page": page_num,
                "table_index": table_idx,
                "dimensions": {
                    "rows": num_rows,
                    "columns": num_cols
                },
                "raw_data": cleaned_table,
                # 🔹 NEW: include per-cell metadata records for easy downstream use
                "cells": cell_records,
                # 🔹 NEW: add header inference summary
                "header_info": header_info,
                "metrics": {
                    "total_cells": total_cells,
                    "non_empty_cells": non_empty_cells,
                    "fill_rate": non_empty_cells / max(total_cells, 1),
                    "confidence": min(0.9, non_empty_cells / max(total_cells, 1))
                },
                "extraction_method": "pdfplumber"
            }
            
            logger.info(
                f"📊 {self.agent_name}: Processed table {num_rows}x{num_cols}, "
                f"fill rate: {table_info['metrics']['fill_rate']:.2f}"
            )
            
            return table_info
            
        except Exception as e:
            logger.warning(f"⚠️ {self.agent_name}: Table processing failed: {e}")
            return None
    
    def _calculate_overall_confidence(self, tables: List[Dict], raw_text: str) -> float:
        """Calculate overall confidence in the extraction"""
        if not tables:
            # Low confidence if no tables found
            text_length = len(raw_text.strip())
            if text_length < 50:
                return 0.1  # Very low - likely image-based PDF
            else:
                return 0.3  # Some text but no tables detected
        
        # Calculate average confidence from all tables
        table_confidences = [
            table.get('metrics', {}).get('confidence', 0.0) 
            for table in tables
        ]
        
        if table_confidences:
            avg_confidence = sum(table_confidences) / len(table_confidences)
            return min(0.95, avg_confidence)
        
        return 0.5


    def _build_cell_records(self, cleaned_table: List[List[str]]) -> (List[Dict[str, Any]], Dict[str, Any]):
        """
        Build a list of per-cell records with value, row/col indices, and inferred headers.
        Returns (cell_records, header_info).
        """
        if not cleaned_table:
            return [], {"header_rows": [], "header_cols": [], "column_headers": {}}

        header_rows = self._identify_header_rows(cleaned_table)
        header_cols = self._identify_header_cols(cleaned_table)
        column_headers = self._extract_column_headers(cleaned_table, header_rows)

        cells: List[Dict[str, Any]] = []
        max_cols = max(len(r) for r in cleaned_table)

        for r, row in enumerate(cleaned_table):
            for c in range(max_cols):
                val = row[c] if c < len(row) else ""
                record = {
                    "value": val,
                    "row_index": r,
                    "column_index": c,
                    "is_header_row": r in header_rows,
                    "is_header_col": c in header_cols,
                    "row_headers": self._extract_row_headers_for_cell(cleaned_table, r, header_cols),
                    "column_headers": column_headers.get(c, [])
                }
                cells.append(record)

        header_info = {
            "header_rows": header_rows,
            "header_cols": header_cols,
            "column_headers": column_headers  # {col_index: [hdr1, hdr2, ...]}
        }
        return cells, header_info

    def _identify_header_rows(self, table: List[List[str]]) -> List[int]:
        """
        Heuristic: if the first row contains mostly text (not empty/numeric-only), treat as header row.
        """
        if not table:
            return []
        first = table[0]
        if not first:
            return []
        text_like = 0
        for cell in first:
            s = (cell or "").strip()
            if not s:
                continue
            # consider numeric-only as not text-like
            if not self._is_number_like(s):
                text_like += 1
        return [0] if text_like >= max(1, len(first) // 2) else []

    def _identify_header_cols(self, table: List[List[str]]) -> List[int]:
        """
        Heuristic: if first column has text on at least half of the rows, treat it as a header column.
        """
        if not table or not any(table):
            return []
        first_col_vals = []
        for row in table:
            if row:
                first_col_vals.append((row[0] or "").strip())
        if not first_col_vals:
            return []
        text_count = sum(1 for s in first_col_vals if s and not self._is_number_like(s))
        return [0] if text_count >= max(1, len(first_col_vals) // 2) else []

    def _extract_column_headers(self, table: List[List[str]], header_rows: List[int]) -> Dict[int, List[str]]:
        """
        Collect header strings per column from the identified header rows.
        """
        headers: Dict[int, List[str]] = {}
        if not header_rows:
            return headers
        max_cols = max(len(r) for r in table)
        for c in range(max_cols):
            col_hdrs: List[str] = []
            for r in header_rows:
                row = table[r]
                if c < len(row):
                    s = (row[c] or "").strip()
                    if s:
                        col_hdrs.append(s)
            if col_hdrs:
                headers[c] = col_hdrs
        return headers

    def _extract_row_headers_for_cell(self, table: List[List[str]], row_idx: int, header_cols: List[int]) -> List[str]:
        """
        Gather text from identified header columns for a given row.
        """
        out: List[str] = []
        if 0 <= row_idx < len(table):
            row = table[row_idx]
            for c in header_cols:
                if c < len(row):
                    s = (row[c] or "").strip()
                    if s:
                        out.append(s)
        return out

    def _is_number_like(self, s: str) -> bool:
        """
        True if string looks like a number (int/float with optional comma separators).
        """
        try:
            float(s.replace(",", ""))
            return True
        except Exception:
            return False
        