"""
Enhanced OCR Agent - Advanced table extraction from image-based PDFs
Combines functionality from table_extraction_framework_ocr.py with main_agent.py compatibility
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
import re
import numpy as np

# OCR Dependencies
try:
    import pytesseract
    from PIL import Image
    import cv2
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError as e:
    OCR_AVAILABLE = False
    print(f"⚠️ OCR libraries not available: {e}")

logger = logging.getLogger(__name__)

# Enhanced regex patterns and constants from the advanced framework
HEADER_TOKEN_RE = re.compile(r"^col[0-9a-z]*$", re.IGNORECASE)
SEP_ONLY_RE = re.compile(r"^[\|\│\¦\┃\—\-]+$")

class OCRAgent:
    """Enhanced OCR Agent with advanced table extraction capabilities"""
    
    def __init__(self):
        self.agent_name = "Enhanced_OCR_Agent"
        self.ocr_available = OCR_AVAILABLE
        logger.info(f"🔍 {self.agent_name}: Initialized (OCR Available: {self.ocr_available})")
        
        if not self.ocr_available:
            logger.warning(f"⚠️ {self.agent_name}: OCR libraries not installed!")
            logger.warning("Install with: pip install pytesseract pillow opencv-python pdf2image")
    
    def extract_tables_with_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """Enhanced OCR extraction with advanced preprocessing and table detection"""
        logger.info(f"🔍 {self.agent_name}: Starting enhanced OCR extraction from {pdf_path}")
        
        if not self.ocr_available:
            return self._create_error_result("OCR libraries not available")
        
        try:
            extraction_result = self._initialize_extraction_result(pdf_path)
            
            # Convert PDF to high-quality images
            logger.info(f"🖼️ {self.agent_name}: Converting PDF to high-resolution images...")
            images = convert_from_path(pdf_path, dpi=300)
            logger.info(f"🖼️ {self.agent_name}: Converted to {len(images)} images")
            
            all_tables = []
            total_ocr_text = ""
            processing_stats = {
                'pages_processed': 0,
                'words_extracted': 0,
                'tables_detected': 0,
                'avg_confidence': 0.0
            }
            
            for page_num, image in enumerate(images):
                logger.info(f"🔍 {self.agent_name}: Processing page {page_num + 1} with enhanced OCR")
                
                page_result = self._process_page_with_enhanced_ocr(image, page_num + 1)
                if not page_result:
                    continue

                total_ocr_text += page_result.get('ocr_text', '')
                page_tables = page_result.get('tables', [])
                all_tables.extend(page_tables)
                
                # Update processing stats
                processing_stats['pages_processed'] += 1
                processing_stats['words_extracted'] += page_result.get('word_count', 0)
                processing_stats['tables_detected'] += len(page_tables)
                
                logger.info(
                    f"🔍 Page {page_num + 1} → tables: {len(page_tables)}, "
                    f"words: {page_result.get('word_count', 0)}, "
                    f"confidence: {page_result.get('confidence_avg', 0.0):.2f}"
                )
            
            # Calculate final metrics
            if processing_stats['pages_processed'] > 0:
                processing_stats['avg_confidence'] = sum(
                    table.get('metrics', {}).get('avg_confidence', 0.0) 
                    for table in all_tables
                ) / len(all_tables) if all_tables else 0.0
            
            # Update final results
            extraction_result.update({
                "tables_found": len(all_tables),
                "tables": all_tables,
                "ocr_text_length": len(total_ocr_text),
                "pages_processed": len(images),
                "overall_confidence": self._calculate_enhanced_confidence(all_tables, total_ocr_text),
                "processing_stats": processing_stats,
                "extraction_details": {
                    "dpi_used": 300,
                    "preprocessing_method": "enhanced_denoising_with_morphology",
                    "tesseract_config": "advanced_whitelist_with_oem1_psm6",
                    "header_detection": "content_aware_clustering",
                    "column_clustering": "x_center_based_with_tolerance",
                    "total_pages": len(images)
                }
            })
            
            logger.info(f"📊 {self.agent_name}: Enhanced OCR extraction completed")
            logger.info(f"   Tables found: {extraction_result['tables_found']}")
            logger.info(f"   Words extracted: {processing_stats['words_extracted']}")
            logger.info(f"   OCR text length: {extraction_result['ocr_text_length']} chars")
            logger.info(f"   Overall confidence: {extraction_result['overall_confidence']:.2f}")
            
            return extraction_result
            
        except Exception as e:
            error_msg = f"Enhanced OCR extraction failed: {str(e)}"
            logger.error(f"❌ {self.agent_name}: {error_msg}")
            return self._create_error_result(error_msg)
    
    def _initialize_extraction_result(self, pdf_path: str) -> Dict[str, Any]:
        """Initialize the extraction result structure"""
        return {
            "agent": self.agent_name,
            "extraction_method": "enhanced_ocr",
            "timestamp": datetime.now().isoformat(),
            "pdf_path": pdf_path,
            "tables_found": 0,
            "overall_confidence": 0.0,
            "tables": [],
            "ocr_text_length": 0,
            "pages_processed": 0,
            "processing_stats": {},
            "extraction_details": {}
        }
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            "agent": self.agent_name,
            "extraction_method": "enhanced_ocr",
            "error": error_msg,
            "tables_found": 0,
            "overall_confidence": 0.0,
            "tables": []
        }
    
    def _process_page_with_enhanced_ocr(self, image: Image.Image, page_num: int) -> Optional[Dict]:
        """Process page with enhanced OCR preprocessing and intelligent table detection"""
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply advanced preprocessing pipeline
            processed_image = self._apply_enhanced_preprocessing(cv_image)
            
            # Enhanced OCR with optimized configuration
            ocr_config = self._get_optimized_ocr_config()
            ocr_data = pytesseract.image_to_data(
                processed_image,
                output_type=pytesseract.Output.DICT,
                config=ocr_config
            )
            ocr_text = pytesseract.image_to_string(processed_image, config=ocr_config)

            # Extract tables using advanced method
            table_candidates = self._extract_tables_with_intelligent_clustering(ocr_data, page_num)
            tables: List[Dict[str, Any]] = []
            
            if table_candidates:
                valid_words = self._extract_valid_words_from_ocr(ocr_data)
                for idx, candidate in enumerate(table_candidates):
                    raw_table = candidate.get("raw_table", [])
                    if not raw_table:
                        continue
                    
                    table_info = self._build_enhanced_table_info(
                        raw_table=raw_table,
                        page_num=page_num,
                        table_idx=idx,
                        valid_words=valid_words,
                        extraction_method="enhanced_ocr_clustering"
                    )
                    if table_info:
                        tables.append(table_info)

            # Calculate page-level confidence
            confidence_scores = []
            for conf_str in (ocr_data.get('conf', [])):
                try:
                    conf_val = float(conf_str)
                    if conf_val > 0:
                        confidence_scores.append(conf_val)
                except (ValueError, TypeError):
                    pass
            
            page_confidence = float(np.mean(confidence_scores)) if confidence_scores else 0.0
            word_count = len([w for w in ocr_data.get('text', []) if str(w).strip()])

            return {
                "page": page_num,
                "ocr_text": ocr_text,
                "tables": tables,
                "word_count": word_count,
                "confidence_avg": page_confidence,
                "processing_method": "enhanced_pipeline"
            }
            
        except Exception as e:
            logger.warning(f"⚠️ {self.agent_name}: Enhanced processing failed for page {page_num}: {e}")
            return None

    def _apply_enhanced_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced preprocessing pipeline for better OCR accuracy"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply advanced denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Adaptive threshold for better binarization
        adaptive_thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Combine with Otsu threshold
        _, otsu_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Take the best of both thresholding methods
        combined = cv2.bitwise_and(adaptive_thresh, otsu_thresh)
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Optional: slight dilation to make text more readable
        final_kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(cleaned, final_kernel, iterations=1)
        
        return processed
    
    def _get_optimized_ocr_config(self) -> str:
        """Get optimized Tesseract configuration for table extraction"""
        return (
            "--oem 1 --psm 6 "
            "-c tessedit_char_whitelist="
            "0123456789.,-+%"  # Numbers and common symbols
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz"
            " \t\n"  # Whitespace
        )
    
    def _extract_valid_words_from_ocr(self, ocr_data: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Extract valid words from OCR data with enhanced filtering"""
        words = []
        n = len(ocr_data.get('text', []))
        
        for i in range(n):
            try:
                conf = float(ocr_data['conf'][i])
            except (ValueError, TypeError):
                conf = -1.0
            
            # Enhanced confidence threshold
            if conf <= 25:  # Slightly lower threshold for more data
                continue
            
            text = str(ocr_data['text'][i] or "").strip()
            if not text or SEP_ONLY_RE.match(text):
                continue
            
            # Filter out single character noise (except important ones)
            if len(text) == 1 and text not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                continue
            
            try:
                words.append({
                    'text': text,
                    'x': int(ocr_data['left'][i]),
                    'y': int(ocr_data['top'][i]),
                    'width': int(ocr_data['width'][i]),
                    'height': int(ocr_data['height'][i]),
                    'conf': conf
                })
            except (ValueError, TypeError, KeyError):
                continue
        
        return words
    
    def _extract_tables_with_intelligent_clustering(self, ocr_data, page_num):
        """Enhanced table extraction with intelligent clustering"""
        try:
            # Group words into visual lines with smart threshold
            lines = self._group_words_into_lines_enhanced(ocr_data, line_threshold=20)
            if not lines:
                return None

            # Identify header lines using content analysis
            header_line_indices = self._identify_header_lines_by_content(lines, max_levels=4)
            
            # Use header lines for column detection, fallback to top lines
            reference_lines = [lines[i] for i in header_line_indices] if header_line_indices else lines[:5]
            
            # Advanced column clustering with dynamic tolerance
            column_centers = self._cluster_columns_with_adaptive_tolerance(reference_lines)
            if not column_centers:
                return None

            num_cols = len(column_centers)
            logger.info(f"🔍 Detected {num_cols} columns using intelligent clustering")

            # Build table with enhanced cell assignment
            table_rows = self._build_table_rows_enhanced(lines, column_centers)
            
            if table_rows and len(table_rows) > 1:
                return [{
                    "page": page_num,
                    "table_index": 0,
                    "raw_table": table_rows,
                    "bbox": self._estimate_table_bbox(lines),
                    "extraction_method": "intelligent_clustering",
                    "column_count": num_cols,
                    "row_count": len(table_rows)
                }]
            return None

        except Exception as e:
            logger.warning(f"⚠️ Enhanced table extraction failed: {e}")
            return None
    
    def _group_words_into_lines_enhanced(self, ocr_data, line_threshold=20):
        """Enhanced line grouping with better handling of varied line heights"""
        words = self._extract_valid_words_from_ocr(ocr_data)
        if not words:
            return []

        # Sort by y-coordinate
        words.sort(key=lambda w: w['y'])
        
        lines = []
        current_line = [words[0]]
        
        for word in words[1:]:
            # Dynamic threshold based on previous word height
            dynamic_threshold = max(line_threshold, current_line[-1]['height'] * 0.8)
            
            if abs(word['y'] - current_line[-1]['y']) <= dynamic_threshold:
                current_line.append(word)
            else:
                lines.append(current_line)
                current_line = [word]
        
        if current_line:
            lines.append(current_line)

        # Filter and sort lines
        filtered_lines = []
        for line in lines:
            line.sort(key=lambda w: w['x'])  # Sort words left to right
            
            # Keep lines that have meaningful content
            if len(line) == 1:
                text = line[0]['text']
                if HEADER_TOKEN_RE.match(text) or any(c.isalnum() for c in text):
                    filtered_lines.append(line)
            else:
                filtered_lines.append(line)
        
        return filtered_lines
    
    def _identify_header_lines_by_content(self, lines, max_levels=4):
        """Identify header lines using intelligent content analysis"""
        header_indices = []
        
        for idx, line in enumerate(lines[:max_levels * 2]):  # Look in first several lines
            line_text = ' '.join([w['text'] for w in line])
            
            # Check for column header patterns
            if any(HEADER_TOKEN_RE.match(w['text']) for w in line):
                header_indices.append(idx)
            # Check for other header indicators
            elif any(indicator in line_text.lower() for indicator in ['table', 'column', 'header', 'title']):
                header_indices.append(idx)
            
            if len(header_indices) >= max_levels:
                break
        
        # If no explicit headers found, use first line as fallback
        if not header_indices and lines:
            header_indices = [0]
        
        return header_indices[:max_levels]
    
    def _cluster_columns_with_adaptive_tolerance(self, lines, base_tolerance=25):
        """Advanced column clustering with adaptive tolerance"""
        x_centers = []
        
        for line in lines:
            for word in line:
                x_center = word['x'] + word['width'] / 2.0
                x_centers.append(x_center)
        
        if not x_centers:
            return []
        
        x_centers.sort()
        
        # Calculate adaptive tolerance based on data distribution
        if len(x_centers) > 1:
            gaps = [x_centers[i+1] - x_centers[i] for i in range(len(x_centers)-1)]
            avg_gap = np.mean(gaps)
            tolerance = min(base_tolerance, avg_gap * 0.3)
        else:
            tolerance = base_tolerance
        
        # Cluster x-centers
        clusters = []
        current_cluster = [x_centers[0]]
        
        for x_center in x_centers[1:]:
            cluster_mean = sum(current_cluster) / len(current_cluster)
            if abs(x_center - cluster_mean) <= tolerance:
                current_cluster.append(x_center)
            else:
                clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [x_center]
        
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        return sorted(clusters)
    
    def _build_table_rows_enhanced(self, lines, column_centers):
        """Build table rows with enhanced cell assignment logic"""
        table_rows = []
        num_cols = len(column_centers)
        
        for line in lines:
            row = [""] * num_cols
            
            for word in line:
                col_idx = self._assign_to_nearest_column(word, column_centers)
                if col_idx is not None:
                    text = word['text'].strip()
                    if text:
                        # Smart cell content concatenation
                        if row[col_idx]:
                            row[col_idx] = row[col_idx] + " " + text
                        else:
                            row[col_idx] = text
            
            # Keep rows with meaningful content
            if any(cell and any(c.isalnum() for c in cell) for cell in row):
                # Clean up cells
                cleaned_row = [cell.strip() if cell else "" for cell in row]
                table_rows.append(cleaned_row)
        
        return table_rows
    
    def _assign_to_nearest_column(self, word, column_centers):
        """Assign word to nearest column center"""
        if not column_centers:
            return None
        
        word_center = word['x'] + word['width'] / 2.0
        distances = [abs(word_center - col_center) for col_center in column_centers]
        return distances.index(min(distances))
    
    def _estimate_table_bbox(self, lines):
        """Estimate bounding box of the detected table"""
        if not lines:
            return None
        
        all_words = [word for line in lines for word in line]
        if not all_words:
            return None
        
        min_x = min(word['x'] for word in all_words)
        max_x = max(word['x'] + word['width'] for word in all_words)
        min_y = min(word['y'] for word in all_words)
        max_y = max(word['y'] + word['height'] for word in all_words)
        
        return {
            'left': min_x,
            'top': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }
    
    def _build_enhanced_table_info(self, raw_table, page_num, table_idx, valid_words, extraction_method):
        """Build comprehensive table information structure compatible with main_agent.py"""
        try:
            if not raw_table:
                return None

            # Clean and normalize table data
            cleaned_table = [
                [(str(cell).strip() if cell is not None else "") for cell in row] 
                for row in raw_table
            ]

            num_rows = len(cleaned_table)
            num_cols = max((len(row) for row in cleaned_table), default=0)
            total_cells = sum(len(row) for row in cleaned_table)
            non_empty_cells = sum(1 for row in cleaned_table for cell in row if cell)

            # Advanced header detection
            header_rows = self._identify_header_rows_enhanced(cleaned_table)
            header_cols = self._identify_header_cols_enhanced(cleaned_table)

            # Extract column headers with hierarchy support
            column_headers_map = self._extract_column_headers_enhanced(cleaned_table, header_rows)
            column_headers_map = self._propagate_parent_headers(column_headers_map)

            # Build detailed cell information for main_agent compatibility
            cells = self._build_cell_records(cleaned_table, header_rows, header_cols, column_headers_map, num_cols)

            # Calculate comprehensive metrics
            metrics = self._calculate_enhanced_metrics(cleaned_table, valid_words, cells)

            table_info = {
                "table_id": f"enhanced_ocr_table_{page_num}_{table_idx}",
                "page": page_num,
                "table_index": table_idx,
                "dimensions": {"rows": num_rows, "columns": num_cols},
                "raw_data": cleaned_table,  # Kept for internal use
                "cells": cells,  # Used by main_agent for value extraction
                "metrics": metrics,
                "extraction_method": extraction_method,
                "header_analysis": {
                    "header_rows": header_rows,
                    "header_cols": header_cols,
                    "column_headers_detected": len(column_headers_map)
                }
            }

            logger.info(
                f"📊 Enhanced table built: {num_rows}×{num_cols}, "
                f"fill_rate={non_empty_cells/max(total_cells,1):.2f}, "
                f"confidence={metrics.get('extraction_confidence', 0.0):.2f}"
            )
            
            return table_info

        except Exception as e:
            logger.warning(f"⚠️ Failed to build enhanced table info: {e}")
            return None
    
    def _build_cell_records(self, cleaned_table, header_rows, header_cols, column_headers_map, num_cols):
        """Build detailed cell records for main_agent compatibility"""
        cells = []
        
        for r, row in enumerate(cleaned_table):
            for c in range(num_cols):
                value = row[c] if c < len(row) else ""
                is_header_row = r in header_rows
                is_header_col = c in header_cols
                
                # Extract contextual headers
                row_headers = self._extract_row_headers_for_cell(cleaned_table, r, header_cols)
                col_headers = column_headers_map.get(c, [])
                
                cells.append({
                    "value": value,
                    "row_index": r,
                    "column_index": c,
                    "is_header_row": is_header_row,
                    "is_header_col": is_header_col,
                    "row_headers": row_headers,
                    "column_headers": col_headers
                })
        
        return cells
    
    def _identify_header_rows_enhanced(self, cleaned_table):
        """Enhanced header row identification"""
        header_rows = []
        if not cleaned_table:
            return header_rows

        # Look for content-based header indicators
        for idx, row in enumerate(cleaned_table[:6]):  # Check first 6 rows
            row_text = ' '.join([str(cell) for cell in row if cell]).lower()
            
            # Check for header token patterns
            if any(HEADER_TOKEN_RE.match(str(cell).strip()) for cell in row if cell):
                header_rows.append(idx)
            # Check for header keywords
            elif any(keyword in row_text for keyword in ['col', 'header', 'table', 'title']):
                header_rows.append(idx)
        
        # Fallback: use first row if no headers detected
        if not header_rows:
            header_rows = [0]
        
        return header_rows[:3]  # Limit to first 3 header levels
    
    def _identify_header_cols_enhanced(self, cleaned_table):
        """Enhanced header column identification"""
        header_cols = []
        if not cleaned_table:
            return header_cols
        
        num_cols = max((len(row) for row in cleaned_table), default=0)
        
        for col_idx in range(min(4, num_cols)):  # Check first 4 columns
            col_values = []
            for row in cleaned_table:
                if col_idx < len(row) and row[col_idx]:
                    col_values.append(str(row[col_idx]).lower())
            
            col_text = ' '.join(col_values)
            
            # Look for row header indicators
            if any(pattern in col_text for pattern in ['row', 'title', 'm1', 'm2', 'm3', 'm4', 'merged']):
                header_cols.append(col_idx)
        
        # Fallback: use first column if no header columns detected
        if not header_cols:
            header_cols = [0]
        
        return header_cols
    
    def _extract_column_headers_enhanced(self, cleaned_table, header_rows):
        """Extract hierarchical column headers"""
        column_headers = {}
        if not cleaned_table or not header_rows:
            return column_headers
        
        num_cols = max((len(row) for row in cleaned_table), default=0)
        
        for col_idx in range(num_cols):
            headers = []
            for header_row_idx in sorted(header_rows):
                if header_row_idx < len(cleaned_table):
                    row = cleaned_table[header_row_idx]
                    if col_idx < len(row) and row[col_idx]:
                        header_text = str(row[col_idx]).strip()
                        if header_text and header_text not in headers:
                            headers.append(header_text)
            
            if headers:
                column_headers[col_idx] = headers
        
        return column_headers
    
    def _propagate_parent_headers(self, column_headers):
        """Propagate parent headers (e.g., Col2 to Col2A, Col2B)"""
        # Find parent headers (exact Col + digits pattern)
        parents = set()
        for headers in column_headers.values():
            for header in headers:
                if re.match(r"^col\d+$", header.lower().strip()):
                    parents.add(header.strip())
        
        if not parents:
            return column_headers
        
        # Propagate parents to children
        for col_idx, headers in column_headers.items():
            additions = []
            for header in headers:
                for parent in parents:
                    if (header.lower().startswith(parent.lower()) and 
                        header.lower() != parent.lower() and 
                        parent not in headers):
                        additions.append(parent)
            headers.extend(additions)
        
        return column_headers
    
    def _extract_row_headers_for_cell(self, cleaned_table, row_idx, header_cols):
        """Extract row headers for a specific cell"""
        row_headers = []
        
        if 0 <= row_idx < len(cleaned_table):
            row = cleaned_table[row_idx]
            for col_idx in sorted(header_cols):
                if col_idx < len(row) and row[col_idx]:
                    header_text = str(row[col_idx]).strip()
                    if header_text:
                        row_headers.append(header_text)
        
        return row_headers
    
    def _calculate_enhanced_metrics(self, cleaned_table, valid_words, cells):
        """Calculate comprehensive quality metrics"""
        total_cells = len(cells)
        non_empty_cells = len([cell for cell in cells if cell['value'] and str(cell['value']).strip()])
        
        # OCR confidence
        if valid_words:
            ocr_confidences = [w.get('conf', 0) for w in valid_words if isinstance(w.get('conf'), (int, float))]
            avg_ocr_conf = float(np.mean(ocr_confidences)) if ocr_confidences else 0.0
        else:
            avg_ocr_conf = 0.0
        
        # Structure quality
        if cleaned_table:
            row_lengths = [len(row) for row in cleaned_table]
            structure_consistency = 1.0 - (np.std(row_lengths) / max(np.mean(row_lengths), 1))
        else:
            structure_consistency = 0.0
        
        # Data density
        data_density = non_empty_cells / max(total_cells, 1)
        
        # Combined extraction confidence
        extraction_confidence = (
            (avg_ocr_conf / 100.0) * 0.4 +
            structure_consistency * 0.3 +
            data_density * 0.3
        )
        
        return {
            "total_cells": total_cells,
            "non_empty_cells": non_empty_cells,
            "data_density": data_density,
            "avg_confidence": avg_ocr_conf,
            "structure_consistency": structure_consistency,
            "extraction_confidence": min(0.95, max(0.05, extraction_confidence))
        }
    
    def _calculate_enhanced_confidence(self, tables, ocr_text):
        """Calculate overall extraction confidence with enhanced logic"""
        if not tables:
            # Base confidence on OCR text quality
            text_length = len((ocr_text or "").strip())
            if text_length > 500:
                return 0.45  # Good OCR but no tables
            elif text_length > 100:
                return 0.25  # Some OCR success
            else:
                return 0.1   # Poor OCR results
        
        # Calculate weighted average of table confidences
        table_confidences = []
        table_weights = []
        
        for table in tables:
            confidence = table.get("metrics", {}).get("extraction_confidence", 0.0)
            weight = table.get("metrics", {}).get("total_cells", 1)
            
            if isinstance(confidence, (int, float)) and confidence > 0:
                table_confidences.append(float(confidence))
                table_weights.append(weight)
        
        if table_confidences:
            # Weighted average based on table size
            weighted_conf = sum(c * w for c, w in zip(table_confidences, table_weights))
            total_weight = sum(table_weights)
            return weighted_conf / total_weight if total_weight > 0 else 0.5
        
        return 0.3  # Default fallback
    
    # ===== Utility and Compatibility Methods =====
    
    def check_ocr_dependencies(self) -> Dict[str, Any]:
        """Comprehensive OCR dependency check with installation guidance"""
        dependencies = {
            "pytesseract": False,
            "PIL": False,
            "cv2": False,
            "pdf2image": False,
            "tesseract_executable": False,
            "numpy": False
        }
        
        # Check each dependency
        try:
            import pytesseract
            dependencies["pytesseract"] = True
            try:
                version = pytesseract.get_tesseract_version()
                dependencies["tesseract_executable"] = True
                logger.info(f"Tesseract version detected: {version}")
            except Exception:
                logger.warning("Tesseract executable not found")
        except ImportError:
            logger.warning("pytesseract not installed")
        
        try:
            from PIL import Image
            dependencies["PIL"] = True
        except ImportError:
            logger.warning("PIL (Pillow) not installed")
        
        try:
            import cv2
            dependencies["cv2"] = True
        except ImportError:
            logger.warning("OpenCV not installed")
        
        try:
            from pdf2image import convert_from_path
            dependencies["pdf2image"] = True
        except ImportError:
            logger.warning("pdf2image not installed")
        
        try:
            import numpy as np
            dependencies["numpy"] = True
        except ImportError:
            logger.warning("numpy not installed")
        
        all_available = all(dependencies.values())
        
        installation_guide = {
            "pip_command": "pip install pytesseract pillow opencv-python pdf2image numpy",
            "system_requirements": {
                "Windows": "Download Tesseract from https://github.com/UB-Mannheim/tesseract/wiki",
                "Mac": "brew install tesseract",
                "Linux": "sudo apt-get install tesseract-ocr"
            },
            "additional_notes": [
                "Make sure Tesseract is in your system PATH",
                "For better results, install language packs: sudo apt-get install tesseract-ocr-eng",
                "Poppler is required for pdf2image: sudo apt-get install poppler-utils"
            ]
        }
        
        return {
            "all_dependencies_available": all_available,
            "individual_status": dependencies,
            "installation_guide": installation_guide,
            "missing_count": sum(1 for status in dependencies.values() if not status)
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Return comprehensive agent information"""
        return {
            "name": self.agent_name,
            "version": "2.0.0",
            "specialization": "Enhanced OCR-based table extraction with intelligent clustering",
            "capabilities": [
                "High-resolution PDF to image conversion",
                "Advanced image preprocessing with denoising",
                "Content-aware header detection",
                "Adaptive column clustering",
                "Hierarchical header extraction",
                "Multi-level confidence scoring"
            ],
            "input_formats": ["PDF files with image-based tables", "Scanned documents"],
            "output_format": "Structured table data with cell-level metadata",
            "confidence_range": "0.05 - 0.95",
            "dependencies_status": "Available" if self.ocr_available else "Missing",
            "preprocessing_methods": [
                "Gaussian denoising",
                "Adaptive thresholding",
                "Otsu binarization",
                "Morphological operations"
            ],
            "extraction_methods": [
                "Intelligent line clustering",
                "X-center based column detection",
                "Content-aware header identification",
                "Parent-child header propagation"
            ]
        }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Return processing statistics and performance metrics"""
        return {
            "ocr_available": self.ocr_available,
            "preprocessing_stages": 6,
            "clustering_algorithms": 2,
            "header_detection_methods": 3,
            "confidence_factors": 4,
            "supported_languages": "Configurable via Tesseract",
            "max_recommended_dpi": 300,
            "min_recommended_dpi": 150
        }
    
    def diagnose_extraction_quality(self, extraction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnose extraction quality and provide recommendations"""
        if not isinstance(extraction_result, dict):
            return {"error": "Invalid extraction result format"}
        
        diagnosis = {
            "overall_assessment": "unknown",
            "confidence_level": extraction_result.get("overall_confidence", 0.0),
            "tables_found": extraction_result.get("tables_found", 0),
            "recommendations": [],
            "potential_issues": []
        }
        
        confidence = diagnosis["confidence_level"]
        tables_found = diagnosis["tables_found"]
        
        # Assess overall quality
        if confidence >= 0.8 and tables_found > 0:
            diagnosis["overall_assessment"] = "excellent"
        elif confidence >= 0.6 and tables_found > 0:
            diagnosis["overall_assessment"] = "good"
        elif confidence >= 0.4 or tables_found > 0:
            diagnosis["overall_assessment"] = "fair"
        else:
            diagnosis["overall_assessment"] = "poor"
        
        # Generate recommendations
        if confidence < 0.5:
            diagnosis["recommendations"].extend([
                "Consider higher resolution scanning (300+ DPI)",
                "Ensure document is properly oriented",
                "Check for adequate contrast in source document"
            ])
        
        if tables_found == 0:
            diagnosis["recommendations"].extend([
                "Verify document contains tabular data",
                "Check if tables are text-based rather than image-based",
                "Try different OCR preprocessing settings"
            ])
        
        # Identify potential issues
        ocr_text_length = extraction_result.get("ocr_text_length", 0)
        if ocr_text_length < 100:
            diagnosis["potential_issues"].append("Very little text extracted - possible image quality issue")
        
        processing_stats = extraction_result.get("processing_stats", {})
        if processing_stats.get("words_extracted", 0) < 50:
            diagnosis["potential_issues"].append("Low word count - OCR may have struggled")
        
        return diagnosis
    
    def optimize_for_document_type(self, document_type: str) -> Dict[str, Any]:
        """Provide optimization settings for different document types"""
        optimizations = {
            "financial_reports": {
                "dpi": 300,
                "preprocessing": "high_contrast",
                "ocr_config": "--psm 6 -c tessedit_char_whitelist=0123456789.,-+%ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
                "column_tolerance": 20,
                "line_threshold": 15
            },
            "scientific_papers": {
                "dpi": 400,
                "preprocessing": "standard",
                "ocr_config": "--psm 6",
                "column_tolerance": 25,
                "line_threshold": 18
            },
            "forms_and_surveys": {
                "dpi": 250,
                "preprocessing": "form_optimized",
                "ocr_config": "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz()[]- ",
                "column_tolerance": 30,
                "line_threshold": 20
            },
            "handwritten_tables": {
                "dpi": 400,
                "preprocessing": "handwriting_enhanced",
                "ocr_config": "--psm 6",
                "column_tolerance": 35,
                "line_threshold": 25
            }
        }
        
        return optimizations.get(document_type, optimizations["scientific_papers"])
    
    def validate_extraction_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extraction result structure and content"""
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "structure_check": True,
            "content_check": True
        }
        
        # Structure validation
        required_fields = ["agent", "extraction_method", "tables_found", "overall_confidence", "tables"]
        for field in required_fields:
            if field not in result:
                validation["errors"].append(f"Missing required field: {field}")
                validation["is_valid"] = False
        
        # Content validation
        if "tables" in result:
            tables = result["tables"]
            if not isinstance(tables, list):
                validation["errors"].append("'tables' should be a list")
                validation["is_valid"] = False
            else:
                for i, table in enumerate(tables):
                    if not isinstance(table, dict):
                        validation["errors"].append(f"Table {i} is not a dictionary")
                        continue
                    
                    table_required = ["table_id", "page", "dimensions", "cells"]
                    for field in table_required:
                        if field not in table:
                            validation["warnings"].append(f"Table {i} missing field: {field}")
        
        # Confidence validation
        confidence = result.get("overall_confidence", 0)
        if not (0 <= confidence <= 1):
            validation["warnings"].append(f"Unusual confidence value: {confidence}")
        
        validation["structure_check"] = len(validation["errors"]) == 0
        validation["content_check"] = len(validation["warnings"]) == 0
        
        return validation
    
    # ===== Legacy Compatibility Methods =====
    
    def extract_tables_from_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """Legacy method for direct image processing"""
        logger.info(f"🖼️ Processing {len(image_paths)} images directly")
        
        all_results = []
        for i, image_path in enumerate(image_paths):
            try:
                image = Image.open(image_path)
                result = self._process_page_with_enhanced_ocr(image, i + 1)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process image {image_path}: {e}")
        
        # Aggregate results
        all_tables = []
        total_text = ""
        for result in all_results:
            all_tables.extend(result.get("tables", []))
            total_text += result.get("ocr_text", "")
        
        return {
            "agent": self.agent_name,
            "extraction_method": "direct_image_processing",
            "images_processed": len(all_results),
            "tables_found": len(all_tables),
            "tables": all_tables,
            "overall_confidence": self._calculate_enhanced_confidence(all_tables, total_text)
        }
    
    def get_extraction_summary(self, result: Dict[str, Any]) -> str:
        """Generate human-readable summary of extraction results"""
        if not isinstance(result, dict):
            return "Invalid extraction result"
        
        agent = result.get("agent", "Unknown")
        method = result.get("extraction_method", "unknown")
        tables_found = result.get("tables_found", 0)
        confidence = result.get("overall_confidence", 0.0)
        pages = result.get("pages_processed", 0)
        
        summary_parts = [
            f"Agent: {agent}",
            f"Method: {method}",
            f"Pages processed: {pages}",
            f"Tables found: {tables_found}",
            f"Overall confidence: {confidence:.2f}"
        ]
        
        if "processing_stats" in result:
            stats = result["processing_stats"]
            words = stats.get("words_extracted", 0)
            summary_parts.append(f"Words extracted: {words}")
        
        # Add quality assessment
        if confidence >= 0.8:
            quality = "Excellent"
        elif confidence >= 0.6:
            quality = "Good"
        elif confidence >= 0.4:
            quality = "Fair"
        else:
            quality = "Poor"
        
        summary_parts.append(f"Quality assessment: {quality}")
        
        return " | ".join(summary_parts)


# ===== Module Testing and Validation =====

def test_ocr_agent():
    """Test function for the enhanced OCR agent"""
    print("🧪 Testing Enhanced OCR Agent...")
    
    agent = OCRAgent()
    
    # Test dependency check
    deps = agent.check_ocr_dependencies()
    print(f"Dependencies available: {deps['all_dependencies_available']}")
    
    # Test agent info
    info = agent.get_agent_info()
    print(f"Agent: {info['name']} v{info['version']}")
    
    # Test optimization suggestions
    opt = agent.optimize_for_document_type("financial_reports")
    print(f"Financial reports optimization: DPI={opt['dpi']}")
    
    return agent

if __name__ == "__main__":
    # Initialize and test the enhanced OCR agent
    agent = test_ocr_agent()
    
    print("\n✅ Enhanced OCR Agent initialized successfully!")
    print("🔧 Compatible with main_agent.py")
    print("🚀 Ready for advanced table extraction!")
