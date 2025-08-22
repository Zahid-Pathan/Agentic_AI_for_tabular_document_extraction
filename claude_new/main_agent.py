# """
# Main AI Agent - Master coordinator for table extraction workflow
# """
# import json
# import logging
# from typing import Dict, List, Any, Optional, Annotated
# from typing_extensions import TypedDict
# import operator
# from langgraph.graph import StateGraph
# from langchain_community.llms import Ollama


# # Import our specialized agents
# from pdf_agent import PDFAgent
# from ocr_agent import OCRAgent
# from validation_agent import ValidationAgent
# from summary_agent import SummaryAgent

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class MasterState(TypedDict):
#     """Master state for the entire extraction workflow"""
#     pdf_path: str
#     extraction_method: Optional[str]  # 'pdfplumber' or 'ocr'
#     raw_extraction_result: Optional[Dict]
#     validated_result: Optional[Dict]
#     summary: Optional[str]
#     final_json: Optional[Dict]
#     error_messages: Annotated[List[str], operator.add]
#     current_step: str
#     agent_recommendations: List[str]

# class MasterAgent:
#     """Master AI Agent that coordinates all specialized agents"""
    
#     def __init__(self, ollama_model: str = "deepseek-r1:1.5b"):
#         """Initialize the master agent with Ollama LLM"""
#         try:
#             self.llm = Ollama(model=ollama_model, temperature=0.1)
#             logger.info(f"Master Agent initialized with Ollama model: {ollama_model}")
#         except Exception as e:
#             logger.warning(f"Failed to initialize Ollama: {e}")
#             self.llm = None
        
#         # Initialize specialized agents
#         self.pdf_agent = PDFAgent()
#         self.ocr_agent = OCRAgent()
#         self.validation_agent = ValidationAgent(ollama_model=ollama_model)
#         self.summary_agent = SummaryAgent(ollama_model=ollama_model)
        
#         self.graph = self._build_master_graph()
    
#     def _build_master_graph(self) -> StateGraph:
#         """Build the master workflow graph"""
        
#         workflow = StateGraph(MasterState)
        
#         # Define nodes
#         workflow.add_node("initialize", self._initialize)
#         workflow.add_node("pdf_extraction", self._delegate_to_pdf_agent)
#         workflow.add_node("ocr_extraction", self._delegate_to_ocr_agent)
#         workflow.add_node("validation", self._delegate_to_validation_agent)
#         workflow.add_node("summarization", self._delegate_to_summary_agent)
#         workflow.add_node("finalization", self._finalize_output)
#         workflow.add_node("error_handler", self._handle_master_errors)
        
#         # Define workflow
#         workflow.set_entry_point("initialize")
        
#         workflow.add_edge("initialize", "pdf_extraction")
        
#         # Decision point: PDF extraction success determines next step
#         workflow.add_conditional_edges(
#             "pdf_extraction",
#             self._decide_extraction_method,
#             {
#                 "pdf_success": "validation",
#                 "need_ocr": "ocr_extraction",
#                 "error": "error_handler"
#             }
#         )
        
#         workflow.add_conditional_edges(
#             "ocr_extraction",
#             self._check_ocr_success,
#             {
#                 "success": "validation",
#                 "error": "error_handler"
#             }
#         )
        
#         workflow.add_conditional_edges(
#             "validation",
#             self._check_validation_success,
#             {
#                 "valid": "summarization",
#                 "invalid": "error_handler"
#             }
#         )
        
#         workflow.add_edge("summarization", "finalization")
#         workflow.add_edge("finalization", "__end__")
#         workflow.add_edge("error_handler", "__end__")
        
#         return workflow.compile()
    
#     def _initialize(self, state: MasterState) -> MasterState:
#         """Initialize the extraction process"""
#         logger.info("🚀 Master Agent: Initializing table extraction workflow")
        
#         # Validate PDF path
#         pdf_path = state['pdf_path']
#         if not pdf_path.endswith('.pdf'):
#             return {
#                 **state,
#                 "error_messages": ["Invalid file format. Please provide a PDF file."],
#                 "current_step": "error"
#             }
        
#         logger.info(f"📄 Target PDF: {pdf_path}")
        
#         return {
#             **state,
#             "current_step": "initialized",
#             "agent_recommendations": ["Starting with PDF analysis..."]
#         }
    
#     def _delegate_to_pdf_agent(self, state: MasterState) -> MasterState:
#         """Delegate to PDF Agent for initial extraction attempt"""
#         logger.info("🔄 Master Agent: Delegating to PDF Agent...")
        
#         try:
#             pdf_result = self.pdf_agent.extract_tables(state['pdf_path'])
            
#             # Analyze PDF agent results
#             tables_found = pdf_result.get('tables_found', 0)
#             confidence = pdf_result.get('overall_confidence', 0.0)
            
#             logger.info(f"📊 PDF Agent Results: {tables_found} tables, confidence: {confidence:.2f}")
            
#             return {
#                 **state,
#                 "raw_extraction_result": pdf_result,
#                 "extraction_method": "pdfplumber",
#                 "current_step": "pdf_extraction_completed"
#             }
            
#         except Exception as e:
#             error_msg = f"PDF Agent delegation failed: {str(e)}"
#             logger.error(error_msg)
#             return {
#                 **state,
#                 "error_messages": [error_msg],
#                 "current_step": "pdf_extraction_failed"
#             }
    
#     def _delegate_to_ocr_agent(self, state: MasterState) -> MasterState:
#         """Delegate to OCR Agent for image-based extraction"""
#         logger.info("🔄 Master Agent: Delegating to OCR Agent...")
        
#         try:
#             ocr_result = self.ocr_agent.extract_tables_with_ocr(state['pdf_path'])
            
#             tables_found = ocr_result.get('tables_found', 0)
#             confidence = ocr_result.get('overall_confidence', 0.0)
            
#             logger.info(f"👁️ OCR Agent Results: {tables_found} tables, confidence: {confidence:.2f}")
            
#             return {
#                 **state,
#                 "raw_extraction_result": ocr_result,
#                 "extraction_method": "ocr",
#                 "current_step": "ocr_extraction_completed"
#             }
            
#         except Exception as e:
#             error_msg = f"OCR Agent delegation failed: {str(e)}"
#             logger.error(error_msg)
#             return {
#                 **state,
#                 "error_messages": [error_msg],
#                 "current_step": "ocr_extraction_failed"
#             }
    
#     def _delegate_to_validation_agent(self, state: MasterState) -> MasterState:
#         """Delegate to Validation Agent"""
#         logger.info("🔄 Master Agent: Delegating to Validation Agent...")
        
#         try:
#             raw_result = state.get('raw_extraction_result')
#             if not raw_result:
#                 return {
#                     **state,
#                     "error_messages": ["No extraction result to validate"],
#                     "current_step": "validation_failed"
#                 }
            
#             validation_result = self.validation_agent.validate_extraction(raw_result)
            
#             logger.info(f"✅ Validation completed with score: {validation_result.get('validation_score', 0.0):.2f}")
            
#             return {
#                 **state,
#                 "validated_result": validation_result,
#                 "current_step": "validation_completed"
#             }
            
#         except Exception as e:
#             error_msg = f"Validation Agent delegation failed: {str(e)}"
#             logger.error(error_msg)
#             return {
#                 **state,
#                 "error_messages": [error_msg],
#                 "current_step": "validation_failed"
#             }
    
#     def _delegate_to_summary_agent(self, state: MasterState) -> MasterState:
#         """Delegate to Summary Agent"""
#         logger.info("🔄 Master Agent: Delegating to Summary Agent...")
        
#         try:
#             validated_result = state.get('validated_result')
#             if not validated_result:
#                 return {
#                     **state,
#                     "error_messages": ["No validated result to summarize"],
#                     "current_step": "summarization_failed"
#                 }
            
#             summary = self.summary_agent.generate_summary(validated_result)
            
#             logger.info("📝 Summary generated successfully")
            
#             return {
#                 **state,
#                 "summary": summary,
#                 "current_step": "summarization_completed"
#             }
            
#         except Exception as e:
#             error_msg = f"Summary Agent delegation failed: {str(e)}"
#             logger.error(error_msg)
#             return {
#                 **state,
#                 "error_messages": [error_msg],
#                 "current_step": "summarization_failed"
#             }
    
#     def _finalize_output(self, state: MasterState) -> MasterState:
#         """Finalize and format the output"""
#         logger.info("🎯 Master Agent: Finalizing output...")
        
#         try:
#             validated_result = state.get('validated_result', {})
#             summary = state.get('summary', "")
            
#             final_output = {
#                 "master_agent_metadata": {
#                     "extraction_method_used": state.get('extraction_method'),
#                     "workflow_completed": True,
#                     "total_agents_involved": 4,
#                     "final_confidence": validated_result.get('validation_score', 0.0)
#                 },
#                 "extraction_summary": summary,
#                 "detailed_results": validated_result,
#                 "agent_recommendations": state.get('agent_recommendations', [])
#             }
            
#             return {
#                 **state,
#                 "final_json": final_output,
#                 "current_step": "completed"
#             }
            
#         except Exception as e:
#             error_msg = f"Finalization failed: {str(e)}"
#             logger.error(error_msg)
#             return {
#                 **state,
#                 "error_messages": [error_msg],
#                 "current_step": "finalization_failed"
#             }
    
#     def _handle_master_errors(self, state: MasterState) -> MasterState:
#         """Handle errors at the master level"""
#         error_messages = state.get('error_messages', [])
#         logger.error(f"❌ Master Agent: Handling errors: {error_messages}")
        
#         return {
#             **state,
#             "current_step": "master_error",
#             "final_json": {
#                 "master_agent_metadata": {
#                     "extraction_method_used": state.get('extraction_method'),
#                     "workflow_completed": False,
#                     "error_occurred": True
#                 },
#                 "errors": error_messages,
#                 "final_state": state.get('current_step')
#             }
#         }
    
#     def _decide_extraction_method(self, state: MasterState) -> str:
#         """Decide whether PDF extraction was successful or OCR is needed"""
#         raw_result = state.get('raw_extraction_result', {})
#         tables_found = raw_result.get('tables_found', 0)
#         confidence = raw_result.get('overall_confidence', 0.0)
        
#         logger.info(f"🤔 Decision point - Tables: {tables_found}, Confidence: {confidence:.2f}")
        
#         if tables_found > 0 and confidence > 0.5:
#             logger.info("✅ PDF extraction successful, proceeding to validation")
#             return "pdf_success"
#         elif state.get('error_messages'):
#             logger.info("❌ PDF extraction failed with errors")
#             return "error"
#         else:
#             logger.info("🔄 PDF extraction found no tables, trying OCR...")
#             return "need_ocr"
    
#     def _check_ocr_success(self, state: MasterState) -> str:
#         """Check if OCR extraction was successful"""
#         raw_result = state.get('raw_extraction_result', {})
#         tables_found = raw_result.get('tables_found', 0)
        
#         if tables_found > 0:
#             logger.info("✅ OCR extraction successful")
#             return "success"
#         else:
#             logger.info("❌ OCR extraction failed")
#             return "error"
    
#     def _check_validation_success(self, state: MasterState) -> str:
#         """Check if validation was successful"""
#         validated_result = state.get('validated_result', {})
#         validation_score = validated_result.get('validation_score', 0.0)
        
#         if validation_score > 0.3:  # Lower threshold for acceptance
#             logger.info(f"✅ Validation successful (score: {validation_score:.2f})")
#             return "valid"
#         else:
#             logger.info(f"❌ Validation failed (score: {validation_score:.2f})")
#             return "invalid"
    
#     def extract_tables(self, pdf_path: str) -> Dict[str, Any]:
#         """Main method to extract tables using multi-agent approach"""
#         logger.info(f"🎯 Master Agent: Starting multi-agent table extraction for: {pdf_path}")
        
#         # Initialize master state
#         initial_state: MasterState = {
#             "pdf_path": pdf_path,
#             "extraction_method": None,
#             "raw_extraction_result": None,
#             "validated_result": None,
#             "summary": None,
#             "final_json": None,
#             "error_messages": [],
#             "current_step": "master_initialized",
#             "agent_recommendations": []
#         }
        
#         try:
#             # Run the master workflow
#             final_state = self.graph.invoke(initial_state)
            
#             # Print summary
#             self._print_execution_summary(final_state)
            
#             # Return final JSON
#             return final_state.get('final_json', {
#                 "error": "Master workflow failed",
#                 "error_messages": final_state.get('error_messages', []),
#                 "final_state": final_state.get('current_step')
#             })
            
#         except Exception as e:
#             logger.error(f"Master workflow execution failed: {e}")
#             return {
#                 "error": f"Master workflow failed: {str(e)}",
#                 "error_messages": [str(e)],
#                 "final_state": "master_workflow_error"
#             }
    
#     def _print_execution_summary(self, final_state: MasterState):
#         """Print a summary of the execution"""
#         print(f"\n{'='*60}")
#         print("🎯 MASTER AGENT EXECUTION SUMMARY")
#         print(f"{'='*60}")
        
#         method = final_state.get('extraction_method', 'unknown')
#         current_step = final_state.get('current_step', 'unknown')
        
#         print(f"📄 PDF File: {final_state.get('pdf_path', 'unknown')}")
#         print(f"🔧 Extraction Method: {method}")
#         print(f"📊 Final Step: {current_step}")
        
#         # Show errors if any
#         errors = final_state.get('error_messages', [])
#         if errors:
#             print(f"❌ Errors Encountered: {len(errors)}")
#             for i, error in enumerate(errors[-3:]):  # Show last 3 errors
#                 print(f"   {i+1}. {error}")
        
#         # Show summary if available
#         summary = final_state.get('summary')
#         if summary:
#             print(f"\n📝 Extraction Summary:")
#             print(f"   {summary}")
        
#         # Show agent recommendations
#         recommendations = final_state.get('agent_recommendations', [])
#         if recommendations:
#             print(f"\n💡 Agent Recommendations:")
#             for rec in recommendations:
#                 print(f"   • {rec}")
        
#         print(f"{'='*60}\n")

# def main():
#     """Main function to run the multi-agent table extraction"""
    
#     print("🚀 Multi-Agent Table Extraction Framework")
#     print("="*50)
    
#     # Get PDF path from user
#     pdf_path = input("📁 Enter the path to your PDF file: ").strip()
    
#     if not pdf_path:
#         pdf_path = "data/example.pdf"  # Default for testing
#         print(f"Using default path: {pdf_path}")
    
#     try:
#         # Initialize master agent
#         master = MasterAgent(ollama_model="llama3.1")
        
#         # Run extraction
#         results = master.extract_tables(pdf_path)
        
#         # Save results
#         output_file = "multi_agent_extraction_results.json"
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
        
#         print(f"💾 Results saved to: {output_file}")
        
#         # Print final JSON (formatted)
#         print(f"\n📋 FINAL JSON OUTPUT:")
#         print("=" * 50)
#         print(json.dumps(results, indent=2))
        
#         return results
        
#     except Exception as e:
#         logger.error(f"Main execution failed: {e}")
#         print(f"❌ Fatal Error: {e}")
#         return {"fatal_error": str(e)}

# if __name__ == "__main__":
#     results = main()









#--------------------

"""
Main AI Agent - Master coordinator for table extraction workflow
"""
import json
import logging
from typing import Dict, List, Any, Optional, Annotated
from typing_extensions import TypedDict
import operator
from langgraph.graph import StateGraph

# Import our specialized agents
from pdf_agent import PDFAgent
from ocr_agent import OCRAgent  # Optional fallback; keep if you want OCR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MasterState(TypedDict):
    """Master state for the entire extraction workflow"""
    pdf_path: str
    extraction_method: Optional[str]  # 'pdfplumber' or 'ocr'
    raw_extraction_result: Optional[Dict]
    final_json: Optional[Dict]
    error_messages: Annotated[List[str], operator.add]
    current_step: str
    agent_recommendations: List[str]

class MasterAgent:
    """Master Agent that coordinates PDF and (optionally) OCR extraction (no LLMs)."""

    def __init__(self):
        self.pdf_agent = PDFAgent()
        # OCR fallback is optional
        try:
            self.ocr_agent = OCRAgent()
            self.ocr_available = True
        except Exception as e:
            logger.warning(f"OCR agent unavailable: {e}")
            self.ocr_agent = None
            self.ocr_available = False

        self.graph = self._build_master_graph()

    def _build_master_graph(self) -> StateGraph:
        """Build the simplified workflow: Initialize → PDF → [OCR] → Finalize."""
        workflow = StateGraph(MasterState)

        workflow.add_node("initialize", self._initialize)
        workflow.add_node("pdf_extraction", self._delegate_to_pdf_agent)
        if self.ocr_available:
            workflow.add_node("ocr_extraction", self._delegate_to_ocr_agent)
        workflow.add_node("finalization", self._finalize_output)
        workflow.add_node("error_handler", self._handle_master_errors)

        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "pdf_extraction")

        workflow.add_conditional_edges(
            "pdf_extraction",
            self._decide_extraction_method,
            {
                "has_tables": "finalization",
                **({"need_ocr": "ocr_extraction"} if self.ocr_available else {}),
                "error": "error_handler",
            }
        )

        if self.ocr_available:
            workflow.add_conditional_edges(
                "ocr_extraction",
                self._check_ocr_success,
                {
                    "has_tables": "finalization",
                    "error": "error_handler",
                }
            )

        workflow.add_edge("finalization", "__end__")
        workflow.add_edge("error_handler", "__end__")

        return workflow.compile()

    def _initialize(self, state: MasterState) -> MasterState:
        """Initialize the extraction process."""
        logger.info("🚀 Master Agent: Initializing table extraction workflow")
        pdf_path = state['pdf_path']
        if not pdf_path.endswith('.pdf'):
            return {
                **state,
                "error_messages": ["Invalid file format. Please provide a PDF file."],
                "current_step": "error",
            }
        logger.info(f"📄 Target PDF: {pdf_path}")
        return {
            **state,
            "current_step": "initialized",
            "agent_recommendations": ["Starting with PDF analysis..."],
        }

    def _delegate_to_pdf_agent(self, state: MasterState) -> MasterState:
        """Run PDF Agent extraction."""
        logger.info("🔄 Master Agent: Delegating to PDF Agent...")
        try:
            pdf_result = self.pdf_agent.extract_tables(state['pdf_path'])
            tables_found = pdf_result.get('tables_found', 0)
            confidence = pdf_result.get('overall_confidence', 0.0)
            logger.info(f"📊 PDF Agent Results: {tables_found} tables, confidence: {confidence:.2f}")
            return {
                **state,
                "raw_extraction_result": pdf_result,
                "extraction_method": "pdfplumber",
                "current_step": "pdf_extraction_completed",
            }
        except Exception as e:
            error_msg = f"PDF Agent delegation failed: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "error_messages": [error_msg],
                "current_step": "pdf_extraction_failed",
            }

    def _delegate_to_ocr_agent(self, state: MasterState) -> MasterState:
        """Run OCR Agent extraction (fallback)."""
        logger.info("🔄 Master Agent: Delegating to OCR Agent...")
        try:
            if not self.ocr_available or self.ocr_agent is None:
                return {
                    **state,
                    "error_messages": ["OCR agent not available."],
                    "current_step": "ocr_extraction_failed",
                }
            ocr_result = self.ocr_agent.extract_tables_with_ocr(state['pdf_path'])
            tables_found = ocr_result.get('tables_found', 0)
            confidence = ocr_result.get('overall_confidence', 0.0)
            logger.info(f"👁️ OCR Agent Results: {tables_found} tables, confidence: {confidence:.2f}")
            return {
                **state,
                "raw_extraction_result": ocr_result,
                "extraction_method": "ocr",
                "current_step": "ocr_extraction_completed",
            }
        except Exception as e:
            error_msg = f"OCR Agent delegation failed: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "error_messages": [error_msg],
                "current_step": "ocr_extraction_failed",
            }

    # -------- Finalization: keep ONLY values + row/column headers --------
    def _finalize_output(self, state: MasterState) -> MasterState:
        """Build final JSON containing only cell values + row/column headers."""
        logger.info("🎯 Master Agent: Finalizing output (values + headers only)")
        try:
            raw = state.get('raw_extraction_result') or {}
            extracted_values = self._collect_value_header_records(raw)

            final_output = {
                "master_agent_metadata": {
                    "extraction_method_used": state.get('extraction_method'),
                    "workflow_completed": True,
                    "total_agents_involved": 2 if self.ocr_available else 1,
                    "final_confidence": raw.get("overall_confidence", 0.0),
                },
                # 👇 ONLY the values with metadata (no raw tables, no full table dumps)
                "extracted_values": extracted_values,
                "agent_recommendations": state.get('agent_recommendations', []),
            }

            return {
                **state,
                "final_json": final_output,
                "current_step": "completed",
            }
        except Exception as e:
            error_msg = f"Finalization failed: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "error_messages": [error_msg],
                "current_step": "finalization_failed",
            }

    def _collect_value_header_records(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        From the agent raw result, collect only non-header cell values and their header metadata.
        The PDFAgent provides table entries with `cells` records:
          { value, row_index, column_index, is_header_row, is_header_col, row_headers[], column_headers[] }
        We keep cells that:
          - are not header row/col cells AND
          - have at least one header present (row_headers or column_headers) AND
          - have a non-empty value.
        """
        out: List[Dict[str, Any]] = []
        for t in raw.get("tables", []):
            table_id = t.get("table_id")
            page = t.get("page")
            table_index = t.get("table_index")

            for rec in t.get("cells", []):
                val = (rec.get("value") or "").strip() if isinstance(rec.get("value"), str) else rec.get("value")
                if not val and val != 0:
                    continue
                if rec.get("is_header_row") or rec.get("is_header_col"):
                    continue
                row_hdrs = rec.get("row_headers", []) or []
                col_hdrs = rec.get("column_headers", []) or []
                if not row_hdrs and not col_hdrs:
                    continue

                out.append({
                    "table_id": table_id,
                    "page": page,
                    "table_index": table_index,
                    "row_index": rec.get("row_index"),
                    "column_index": rec.get("column_index"),
                    "value": val,
                    "row_headers": row_hdrs,
                    "column_headers": col_hdrs,
                })
        return out

    def _handle_master_errors(self, state: MasterState) -> MasterState:
        """Handle errors at the master level."""
        error_messages = state.get('error_messages', [])
        logger.error(f"❌ Master Agent: Handling errors: {error_messages}")
        return {
            **state,
            "current_step": "master_error",
            "final_json": {
                "master_agent_metadata": {
                    "extraction_method_used": state.get('extraction_method'),
                    "workflow_completed": False,
                    "error_occurred": True,
                },
                "errors": error_messages,
                "final_state": state.get('current_step'),
            },
        }

    # -------- Decision helpers --------
    def _decide_extraction_method(self, state: MasterState) -> str:
        """Finalize if PDF found any tables; else try OCR (if available) or error."""
        raw_result = state.get('raw_extraction_result', {}) or {}
        tables_found = raw_result.get('tables_found', 0)
        logger.info(f"\n\n🤔 Decision point (PDF): tables_found={tables_found}")

        if tables_found > 0:
            logger.info("✅ PDF extraction has tables — finalizing.")
            return "has_tables"
        if self.ocr_available:
            logger.info("🔄 No tables found via PDF — trying OCR...")
            return "need_ocr"
        logger.info("❌ No tables found and OCR not available.")
        return "error"

    def _check_ocr_success(self, state: MasterState) -> str:
        """Finalize if OCR found any tables; else error."""
        raw_result = state.get('raw_extraction_result', {}) or {}
        tables_found = raw_result.get('tables_found', 0)
        logger.info(f"🤔 Decision point (OCR): tables_found={tables_found}")
        return "has_tables" if tables_found > 0 else "error"

    # -------- Public API --------
    def extract_tables(self, pdf_path: str) -> Dict[str, Any]:
        """Run the simplified workflow and return the final JSON."""
        logger.info(f"🎯 Master Agent: Starting extraction for: {pdf_path}")
        initial_state: MasterState = {
            "pdf_path": pdf_path,
            "extraction_method": None,
            "raw_extraction_result": None,
            "final_json": None,
            "error_messages": [],
            "current_step": "master_initialized",
            "agent_recommendations": [],
        }
        try:
            final_state = self.graph.invoke(initial_state)
            self._print_execution_summary(final_state)
            return final_state.get('final_json', {
                "error": "Master workflow failed",
                "error_messages": final_state.get('error_messages', []),
                "final_state": final_state.get('current_step'),
            })
        except Exception as e:
            logger.error(f"Master workflow execution failed: {e}")
            return {
                "error": f"Master workflow failed: {str(e)}",
                "error_messages": [str(e)],
                "final_state": "master_workflow_error",
            }

    def _print_execution_summary(self, final_state: MasterState):
        """Print a short summary (values + headers only in final JSON)."""
        print(f"\n{'='*60}")
        print("🎯 MASTER AGENT EXECUTION SUMMARY")
        print(f"{'='*60}")

        method = final_state.get('extraction_method', 'unknown')
        current_step = final_state.get('current_step', 'unknown')

        print(f"📄 PDF File: {final_state.get('pdf_path', 'unknown')}")
        print(f"🔧 Extraction Method: {method}")
        print(f"📊 Final Step: {current_step}")

        errors = final_state.get('error_messages', [])
        if errors:
            print(f"❌ Errors Encountered: {len(errors)}")
            for i, error in enumerate(errors[-3:]):
                print(f"   {i+1}. {error}")

        # Quick count of extracted value records
        fj = final_state.get("final_json") or {}
        values = fj.get("extracted_values", []) if isinstance(fj, dict) else []
        print(f"✅ Extracted value records: {len(values)}")
        print(f"{'='*60}\n")


def main():
    """Run the simplified extraction and save only values+headers to JSON."""
    print("🚀 Table Extraction (PDF → [OCR] → JSON: values+headers only)")
    print("="*50)

    pdf_path = input("📁 Enter the path to your PDF file: ").strip()
    if not pdf_path:
        pdf_path = "data/example.pdf"  # Default for testing
        print(f"Using default path: {pdf_path}")

    try:
        master = MasterAgent()
        results = master.extract_tables(pdf_path)

        output_file = "claude_new/multi_agent_extraction_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Results saved to: {output_file}")
        print("\n📋 FINAL JSON OUTPUT (values + headers only):")
        print("=" * 50)
        print(json.dumps(results, indent=2))
        return results

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        print(f"❌ Fatal Error: {e}")
        return {"fatal_error": str(e)}

if __name__ == "__main__":
    results = main()
