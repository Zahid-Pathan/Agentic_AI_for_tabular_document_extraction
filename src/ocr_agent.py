"""
Enhanced OCR Agent - Drop-in replacement for original OCR agent with better character recognition
Fully compatible with main_agent.py workflow
"""
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os
import re
import numpy as np
from collections import Counter, defaultdict

# OCR Dependencies
try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    import cv2
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError as e:
    OCR_AVAILABLE = False
    print(f"⚠️ OCR libraries not available: {e}")

logger = logging.getLogger(__name__)

class OCRAgent:
    """Enhanced OCR Agent with improved character recognition - Drop-in replacement"""
    
    def __init__(self):
        self.agent_name = "Enhanced_OCR_Agent"
        self.ocr_available = OCR_AVAILABLE
        self.language = "eng+deu"  # Default language for Tesseract
        logger.info(f"🔍 {self.agent_name}: Initialized (OCR Available: {self.ocr_available})")
        
        # Enhanced noise patterns for better filtering
        self.noise_patterns = [
            re.compile(r"^[\|\│\¦\┃\─\-\|]+$"),  # Separator lines
            re.compile(r"^[_\-=\*]{2,}$"),        # Underlines/dividers
            re.compile(r"^[\s\.\,\:\;]{1,3}$"),   # Small punctuation
            re.compile(r"^[\[\]{}()]{1,2}$"),     # Brackets only
            re.compile(r"^[\.]{2,}$"),            # Multiple dots
        ]
    
    def extract_tables_with_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """Main extraction method compatible with main_agent.py"""
        logger.info(f"🔍 {self.agent_name}: Starting enhanced OCR extraction from {pdf_path}")
        
        if not self.ocr_available:
            return self._create_error_result("OCR libraries not available")
        
        try:
            extraction_result = self._initialize_extraction_result(pdf_path)
            
            # Convert PDF with enhanced settings
            logger.info(f"🖼️ {self.agent_name}: Converting PDF with enhanced settings...")
            images = self._convert_pdf_enhanced(pdf_path)
            logger.info(f"🖼️ {self.agent_name}: Converted to {len(images)} images")
            
            all_tables = []
            processing_stats = {
                'pages_processed': 0,
                'words_extracted': 0,
                'tables_detected': 0,
                'confidence_scores': []
            }
            
            for page_num, image in enumerate(images):
                logger.info(f"📄 Processing page {page_num + 1} with enhanced OCR")
                
                page_result = self._process_page_enhanced(image, page_num + 1)
                if not page_result:
                    continue

                page_tables = page_result.get('tables', [])
                all_tables.extend(page_tables)
                
                # Update stats
                processing_stats['pages_processed'] += 1
                processing_stats['words_extracted'] += page_result.get('word_count', 0)
                processing_stats['tables_detected'] += len(page_tables)
                if 'confidence_avg' in page_result:
                    processing_stats['confidence_scores'].append(page_result['confidence_avg'])
                
                logger.info(f"✅ Page {page_num + 1} → tables: {len(page_tables)}")
            
            # Calculate final confidence
            overall_confidence = self._calculate_enhanced_confidence(
                all_tables, 
                processing_stats['confidence_scores']
            )
            
            extraction_result.update({
                "tables_found": len(all_tables),
                "tables": all_tables,
                "pages_processed": len(images),
                "overall_confidence": overall_confidence,
                "processing_stats": processing_stats,
                "extraction_method": "enhanced_ocr"
            })
            
            logger.info(f"📊 Enhanced OCR completed: {len(all_tables)} tables, confidence: {overall_confidence:.2f}")
            return extraction_result
            
        except Exception as e:
            error_msg = f"Enhanced OCR extraction failed: {str(e)}"
            logger.error(f"❌ {self.agent_name}: {error_msg}")
            return self._create_error_result(error_msg)
    
    def _convert_pdf_enhanced(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF with enhanced settings for better OCR"""
        try:
            # Try progressively higher DPI for better character recognition
            dpi_levels = [450, 400, 350, 300]  # Start very high
            
            for dpi in dpi_levels:
                try:
                    logger.info(f"📈 Trying {dpi} DPI conversion...")
                    images = convert_from_path(pdf_path, dpi=dpi)
                    
                    if images:
                        # Quick quality check on first image
                        test_quality = self._assess_image_quality_enhanced(images[0])
                        logger.info(f"🔍 Image quality at {dpi} DPI: {test_quality:.3f}")
                        
                        # Accept if good quality or last attempt
                        if test_quality > 0.6 or dpi == dpi_levels[-1]:
                            return images
                        
                except Exception as e:
                    logger.warning(f"DPI {dpi} conversion failed: {e}")
                    continue
            
            # Final fallback
            return convert_from_path(pdf_path, dpi=300)
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            return []
    
    def _assess_image_quality_enhanced(self, image: Image.Image) -> float:
        """Enhanced image quality assessment"""
        try:
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Multiple quality metrics
            # 1. Sharpness (Laplacian variance)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, sharpness / 1500.0)
            
            # 2. Contrast (standard deviation)
            contrast = np.std(gray)
            contrast_score = min(1.0, contrast / 80.0)
            
            # 3. Text-like edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            text_score = min(1.0, edge_density * 20)
            
            # Combined score
            overall_quality = (sharpness_score * 0.4 + contrast_score * 0.3 + text_score * 0.3)
            return overall_quality
            
        except Exception:
            return 0.5
    
    def _process_page_enhanced(self, image: Image.Image, page_num: int) -> Optional[Dict]:
        """Enhanced page processing with better preprocessing"""
        try:
            # Convert to CV2 format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Enhanced preprocessing pipeline
            processed_image = self._preprocess_for_ocr_enhanced(cv_image)
            
            # Multi-mode OCR extraction
            ocr_data = self._extract_ocr_data_enhanced(processed_image)
            
            if not ocr_data:
                return None
            
            # Enhanced table detection
            tables = self._detect_tables_enhanced(ocr_data, page_num)
            
            # Calculate page confidence
            confidence_scores = [float(conf) for conf in ocr_data.get('conf', []) 
                               if str(conf).replace('.','').replace('-','').isdigit() and float(conf) > 0]
            page_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            word_count = len([w for w in ocr_data.get('text', []) if str(w).strip()])
            
            return {
                "page": page_num,
                "tables": tables,
                "word_count": word_count,
                "confidence_avg": page_confidence,
                "processing_method": "enhanced_ocr"
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced processing failed for page {page_num}: {e}")
            return None
    
    def _preprocess_for_ocr_enhanced(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing specifically for table OCR"""
        
        # Convert to PIL for advanced preprocessing
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 1. Enhance contrast and sharpness
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.3)  # Boost contrast
        
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.8)  # Sharpen text
        
        # 2. Smooth and sharpen
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=0.4))
        pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        
        # Convert back to CV2
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # 3. Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image.copy()
        
        # 4. Advanced denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)
        
        # 5. Remove table borders that confuse OCR
        # Detect and remove horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        horizontal_lines = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect and remove vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        vertical_lines = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine line masks and remove from image
        table_lines = cv2.add(horizontal_lines, vertical_lines)
        result = denoised.copy()
        result[table_lines > 50] = 255  # Fill lines with white
        
        # 6. Enhance text thickness for better recognition
        text_kernel = np.ones((2, 2), np.uint8)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, text_kernel)
        
        # 7. Final adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(
            result, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 6
        )
        
        # 8. Clean small noise artifacts
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            255 - adaptive_thresh, connectivity=8
        )
        
        # Remove very small components (likely noise)
        min_size = 15
        cleaned = adaptive_thresh.copy()
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                cleaned[labels == i] = 255
        
        return cleaned
    
    def _extract_ocr_data_enhanced(self, processed_image: np.ndarray) -> Dict[str, List[Any]]:
        """Extract OCR data with enhanced configuration for tables"""
        
        # OCR configurations optimized for table recognition
        ocr_configs = [
            # Best for tables: uniform text block with character whitelist
            f"--oem 3 --psm 6 -l {self.language} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,%-+/()[]{{}}:; -c preserve_interword_spaces=1",

            # Good for mixed content
            f"--oem 3 --psm 3 -l {self.language} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,%-+/()[]{{}}:;",

            # Single text line mode
            f"--oem 3 --psm 8 -l {self.language} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,%-+/()[]{{}}:;",

            # Fallback: general document
            f"--oem 3 --psm 4 -l {self.language}",
        ]
        
        best_result = None
        best_score = 0
        
        for config in ocr_configs:
            try:
                ocr_data = pytesseract.image_to_data(
                    processed_image,
                    output_type=pytesseract.Output.DICT,
                    config=config
                )
                
                # Score this result based on confidence and meaningful content
                score = self._score_ocr_result(ocr_data)
                
                if score > best_score:
                    best_score = score
                    best_result = ocr_data
                    
            except Exception as e:
                logger.warning(f"OCR config failed: {e}")
                continue
        
        logger.info(f"🎯 Best OCR result score: {best_score:.2f}")
        return best_result if best_result else {}
    
    def _score_ocr_result(self, ocr_data: Dict) -> float:
        """Score an OCR result based on confidence and content quality"""
        if not ocr_data or not ocr_data.get('text'):
            return 0.0
        
        total_score = 0
        total_weight = 0
        
        for i, (text, conf) in enumerate(zip(ocr_data.get('text', []), ocr_data.get('conf', []))):
            try:
                confidence = float(conf)
                if confidence <= 0:
                    continue
                
                text_str = str(text).strip()
                if not text_str or self._is_noise_enhanced(text_str):
                    continue
                
                # Weight by text length and content quality
                content_quality = self._calculate_content_quality(text_str)
                weight = len(text_str) * content_quality
                
                total_score += confidence * weight
                total_weight += weight
                
            except (ValueError, TypeError):
                continue
        
        return total_score / max(total_weight, 1) if total_weight > 0 else 0
    
    def _detect_tables_enhanced(self, ocr_data: Dict, page_num: int) -> List[Dict[str, Any]]:
        """Enhanced table detection compatible with main_agent.py format"""
        try:
            # Extract clean words with enhanced filtering
            words = self._extract_clean_words_enhanced(ocr_data)
            if len(words) < 4:  # Need minimum words for a table
                return []
            
            # Group words into lines
            lines = self._group_words_into_lines(words)
            if len(lines) < 2:
                return []
            
            # Build table structure
            table_info = self._build_table_compatible_format(lines, page_num)
            
            return [table_info] if table_info else []
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced table detection failed: {e}")
            return []
    
    def _extract_clean_words_enhanced(self, ocr_data: Dict) -> List[Dict[str, Any]]:
        """Extract and clean words with enhanced filtering"""
        words = []
        n = len(ocr_data.get('text', []))
        
        for i in range(n):
            try:
                conf = float(ocr_data['conf'][i])
            except (ValueError, TypeError):
                conf = 0
            
            # Higher confidence threshold for better accuracy
            if conf < 45:  # Raised from 20 to 45
                continue
            
            text = str(ocr_data['text'][i] or "").strip()
            if not text or self._is_noise_enhanced(text):
                continue
            
            # Enhanced quality scoring
            quality_score = self._calculate_enhanced_quality(text, conf)
            if quality_score < 0.4:  # Higher threshold
                continue
            
            try:
                words.append({
                    'text': text,
                    'x': int(ocr_data['left'][i]),
                    'y': int(ocr_data['top'][i]),
                    'width': int(ocr_data['width'][i]),
                    'height': int(ocr_data['height'][i]),
                    'conf': conf,
                    'quality': quality_score
                })
            except (ValueError, TypeError, KeyError):
                continue
        
        return words
    
    def _is_noise_enhanced(self, text: str) -> bool:
        """Enhanced noise detection"""
        # Check against known noise patterns
        for pattern in self.noise_patterns:
            if pattern.match(text):
                return True
        
        # Additional enhanced checks
        # Too many special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum()) / max(len(text), 1)
        if special_char_ratio > 0.5:
            return True
        
        # Repeated single characters (OCR errors)
        if len(set(text.lower())) == 1 and len(text) > 2:
            return True
        
        # Very short unclear text
        if len(text) <= 2 and not text.isdigit() and not text.isalpha():
            return True
        
        # Common OCR garbage patterns
        garbage_patterns = [r'^[^\w\s]+$', r'^\W{3,}$', r'^[\|]{2,}$']
        for pattern in garbage_patterns:
            if re.match(pattern, text):
                return True
        
        return False
    
    def _calculate_enhanced_quality(self, text: str, confidence: float) -> float:
        """Enhanced quality calculation"""
        quality = confidence / 100.0  # Base on OCR confidence
        
        # Length bonus (capped)
        length_bonus = min(0.15, len(text) * 0.02)
        quality += length_bonus
        
        # Content type bonuses
        if re.search(r'\d', text):  # Contains numbers
            quality += 0.12
        
        if re.search(r'^[A-Za-z]+$', text):  # Pure alphabetic
            quality += 0.08
        
        if re.search(r'^\d+(\.\d+)?$', text):  # Pure numeric
            quality += 0.15
        
        if re.search(r'^[A-Z][a-z]+$', text):  # Proper case
            quality += 0.12
        
        # Table content patterns
        if re.search(r'^\d+\.\d+$', text):  # Decimal numbers
            quality += 0.18
        
        if re.search(r'^\d+\%$', text):  # Percentages
            quality += 0.15
        
        # Penalty for suspicious patterns
        if re.search(r'[^\w\s\.\,\-\+\%\(\)\[\]\/\:]', text):
            quality -= 0.15
        
        return max(0.0, min(1.0, quality))
    
    def _calculate_content_quality(self, text: str) -> float:
        """Calculate content quality for scoring"""
        if not text:
            return 0.0
        
        quality = 0.5  # Base quality
        
        # Bonus for meaningful content
        if re.search(r'\d', text):
            quality += 0.2
        if len(text) >= 3:
            quality += 0.2
        if re.search(r'^[A-Za-z]+$', text):
            quality += 0.15
        
        # Penalty for suspicious patterns
        if len(set(text)) == 1:  # All same character
            quality -= 0.4
        
        return max(0.1, min(1.0, quality))
    
    def _group_words_into_lines(self, words: List[Dict]) -> List[List[Dict]]:
        """Group words into lines using Y-coordinate clustering"""
        if not words:
            return []
        
        # Sort by Y coordinate
        words_sorted = sorted(words, key=lambda w: w['y'])
        
        # Calculate dynamic line threshold
        heights = [w['height'] for w in words_sorted]
        avg_height = np.mean(heights) if heights else 20
        line_threshold = max(8, avg_height * 0.6)
        
        # Group into lines
        lines = []
        current_line = [words_sorted[0]]
        
        for word in words_sorted[1:]:
            y_diff = abs(word['y'] - current_line[-1]['y'])
            
            if y_diff <= line_threshold:
                current_line.append(word)
            else:
                # Sort current line by X coordinate and add to lines
                current_line.sort(key=lambda w: w['x'])
                lines.append(current_line)
                current_line = [word]
        
        if current_line:
            current_line.sort(key=lambda w: w['x'])
            lines.append(current_line)
        
        return lines
    
    def _build_table_compatible_format(self, lines: List[List[Dict]], page_num: int) -> Optional[Dict[str, Any]]:
        """Build table info in format compatible with main_agent.py"""
        try:
            if len(lines) < 2:
                return None
            
            # Determine table dimensions
            num_rows = len(lines)
            max_cols = max(len(line) for line in lines) if lines else 0
            
            if max_cols < 2:  # Need at least 2 columns
                return None
            
            # Build raw table data
            raw_table = []
            for line in lines:
                row_data = [word['text'] for word in line]
                # Pad to max columns
                while len(row_data) < max_cols:
                    row_data.append("")
                raw_table.append(row_data[:max_cols])
            
            # Detect headers (first row and first column as defaults)
            header_rows = [0] if num_rows > 1 else []
            header_cols = [0] if max_cols > 1 else []
            
            # Build column headers map
            column_headers_map = {}
            if header_rows and len(raw_table) > 0:
                for col_idx in range(max_cols):
                    headers = []
                    for header_row_idx in header_rows:
                        if header_row_idx < len(raw_table):
                            header_text = raw_table[header_row_idx][col_idx] if col_idx < len(raw_table[header_row_idx]) else ""
                            if header_text.strip():
                                headers.append(header_text.strip())
                    if headers:
                        column_headers_map[col_idx] = headers
            
            # Build cell records compatible with main_agent.py
            cells = []
            for row_idx, row in enumerate(raw_table):
                for col_idx, value in enumerate(row):
                    if not value.strip():
                        continue
                    
                    # Determine if this is a header cell
                    is_header_row = row_idx in header_rows
                    is_header_col = col_idx in header_cols
                    
                    # Extract row headers (from header columns)
                    row_headers = []
                    for hcol in header_cols:
                        if hcol < len(row) and row[hcol].strip():
                            row_headers.append(row[hcol].strip())
                    
                    # Get column headers
                    col_headers = column_headers_map.get(col_idx, [])
                    
                    cells.append({
                        "value": value.strip(),
                        "row_index": row_idx,
                        "column_index": col_idx,
                        "is_header_row": is_header_row,
                        "is_header_col": is_header_col,
                        "row_headers": row_headers,
                        "column_headers": col_headers
                    })
            
            # Calculate metrics
            total_cells = num_rows * max_cols
            non_empty_cells = len(cells)
            data_density = non_empty_cells / max(total_cells, 1)
            
            # Calculate average confidence from words
            all_words = [word for line in lines for word in line]
            avg_confidence = np.mean([w['conf'] for w in all_words]) / 100.0 if all_words else 0.0
            
            table_info = {
                "table_id": f"enhanced_ocr_table_{page_num}_0",
                "page": page_num,
                "table_index": 0,
                "dimensions": {"rows": num_rows, "columns": max_cols},
                "raw_data": raw_table,
                "cells": cells,
                "metrics": {
                    "total_cells": total_cells,
                    "non_empty_cells": non_empty_cells,
                    "data_density": data_density,
                    "extraction_confidence": avg_confidence,
                    "detection_method": "enhanced_ocr"
                },
                "extraction_method": "enhanced_ocr",
                "header_analysis": {
                    "header_rows": header_rows,
                    "header_cols": header_cols,
                    "column_headers_detected": len(column_headers_map)
                }
            }
            
            return table_info
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to build compatible table format: {e}")
            return None
    
    def _calculate_enhanced_confidence(self, tables: List[Dict], confidence_scores: List[float]) -> float:
        """Calculate overall confidence for the extraction"""
        if not tables:
            return 0.05
        
        # Base confidence on OCR scores
        ocr_confidence = np.mean(confidence_scores) / 100.0 if confidence_scores else 0.3
        
        # Factor in table quality
        table_qualities = []
        for table in tables:
            metrics = table.get('metrics', {})
            data_density = metrics.get('data_density', 0)
            extraction_conf = metrics.get('extraction_confidence', 0.3)
            table_quality = (data_density + extraction_conf) / 2
            table_qualities.append(table_quality)
        
        overall_table_quality = np.mean(table_qualities) if table_qualities else 0.2
        
        # Combine OCR and table quality
        final_confidence = (ocr_confidence * 0.7 + overall_table_quality * 0.3)
        
        # Ensure reasonable bounds
        return max(0.01, min(0.98, final_confidence))
    
    def _initialize_extraction_result(self, pdf_path: str) -> Dict[str, Any]:
        """Initialize extraction result structure"""
        return {
            "agent": self.agent_name,
            "extraction_method": "enhanced_ocr",
            "timestamp": datetime.now().isoformat(),
            "pdf_path": pdf_path,
            "tables_found": 0,
            "overall_confidence": 0.0,
            "tables": [],
            "pages_processed": 0,
            "processing_stats": {}
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