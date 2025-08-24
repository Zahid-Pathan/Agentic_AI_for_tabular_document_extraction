"""
Enhanced Main AI Agent - Master coordinator with Hugging Face LLM and tool calling
"""
import json
import logging
from typing import Dict, List, Any, Optional, Annotated
from typing_extensions import TypedDict
import operator
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Import our specialized agents
from pdf_agent import PDFAgent
from ocr_agent import OCRAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MasterState(TypedDict):
    """Enhanced master state with LLM conversation history"""
    pdf_path: str
    extraction_method: Optional[str]
    raw_extraction_result: Optional[Dict]
    final_json: Optional[Dict]
    error_messages: Annotated[List[str], operator.add]
    current_step: str
    agent_recommendations: List[str]
    messages: Annotated[List, operator.add]
    llm_analysis: Optional[str]
    decision_reasoning: Optional[str]

# Define tools for the LLM to use
@tool
def analyze_extraction_results(extraction_data: Dict[str, Any]) -> str:
    """
    Analyze extraction results and provide quality assessment.
    
    Args:
        extraction_data: The extraction result from PDF or OCR agent
    
    Returns:
        Analysis summary as string
    """
    if not extraction_data:
        return "No extraction data to analyze"
    
    tables_found = extraction_data.get('tables_found', 0)
    confidence = extraction_data.get('overall_confidence', 0.0)
    method = extraction_data.get('extraction_method', 'unknown')
    
    analysis = f"Extraction Analysis:\n"
    analysis += f"- Method: {method}\n"
    analysis += f"- Tables found: {tables_found}\n" 
    analysis += f"- Confidence: {confidence:.2f}\n"
    
    # Analyze individual tables
    tables = extraction_data.get('tables', [])
    if tables:
        for i, table in enumerate(tables[:3]):  # Analyze first 3 tables
            dims = table.get('dimensions', {})
            rows = dims.get('rows', 0)
            cols = dims.get('columns', 0)
            cells = len(table.get('cells', []))
            analysis += f"- Table {i+1}: {rows}x{cols}, {cells} data cells\n"
    
    return analysis

@tool
def make_extraction_decision(analysis_summary: str, tables_found: int, confidence: float) -> str:
    """
    Make a decision about extraction quality and next steps.
    
    Args:
        analysis_summary: Summary of extraction analysis
        tables_found: Number of tables found
        confidence: Confidence score
        
    Returns:
        Decision and reasoning
    """
    if tables_found > 0 and confidence > 0.5:
        return f"SUCCESS: Good extraction quality. Found {tables_found} tables with {confidence:.2f} confidence. Proceed to finalization."
    elif tables_found > 0 and confidence > 0.3:
        return f"ACCEPTABLE: Moderate extraction quality. Found {tables_found} tables with {confidence:.2f} confidence. Proceed with caution."
    elif tables_found == 0:
        return f"NO_TABLES: No tables detected. Consider OCR fallback if available."
    else:
        return f"POOR_QUALITY: Low confidence {confidence:.2f}. Consider retry with different method."

@tool
def generate_final_summary(extraction_data: Dict[str, Any]) -> str:
    """
    Generate a comprehensive summary of the extraction process.
    
    Args:
        extraction_data: Complete extraction results
        
    Returns:
        Final summary string
    """
    if not extraction_data:
        return "No data available for summary."
    
    method = extraction_data.get('extraction_method', 'unknown')
    tables_found = extraction_data.get('tables_found', 0)
    
    summary = f"Table Extraction Summary:\n"
    summary += f"✓ Method used: {method.upper()}\n"
    summary += f"✓ Tables extracted: {tables_found}\n"
    
    if tables_found > 0:
        total_values = 0
        tables = extraction_data.get('tables', [])
        for table in tables:
            cells = table.get('cells', [])
            data_cells = [c for c in cells if not c.get('is_header_row') and not c.get('is_header_col')]
            total_values += len(data_cells)
        
        summary += f"✓ Data values extracted: {total_values}\n"
        summary += f"✓ Extraction successful!"
    else:
        summary += f"✗ No tables found in document"
    
    return summary

class EnhancedMasterAgent:
    """Enhanced Master Agent with Hugging Face LLM and tool calling"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        """Initialize with Hugging Face LLM"""
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
        # Initialize specialized agents
        self.pdf_agent = PDFAgent()
        try:
            self.ocr_agent = OCRAgent()
            self.ocr_available = True
        except Exception as e:
            logger.warning(f"OCR agent unavailable: {e}")
            self.ocr_agent = None
            self.ocr_available = False
        
        # Define tools
        self.tools = [analyze_extraction_results, make_extraction_decision, generate_final_summary]
        self.tool_node = ToolNode(self.tools)
        
        self.graph = self._build_enhanced_graph()
    
    def _initialize_llm(self):
        """Initialize Hugging Face LLM pipeline"""
        try:
            logger.info(f"🤖 Initializing Hugging Face model: {self.model_name}")
            
            # Use a lightweight model optimized for reasoning
            model_name = "microsoft/DialoGPT-small"  # Small, fast model
            
            # Alternative options (uncomment to try):
            # model_name = "distilbert/distilgpt2"  # Very lightweight
            # model_name = "gpt2"  # Standard GPT-2
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("✅ LLM initialized successfully")
            return llm
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM: {e}")
            logger.info("🔄 Falling back to rule-based decisions")
            return None
    
    def _build_enhanced_graph(self) -> StateGraph:
        """Build enhanced workflow with LLM integration"""
        workflow = StateGraph(MasterState)
        
        # Define nodes
        workflow.add_node("initialize", self._initialize)
        workflow.add_node("pdf_extraction", self._delegate_to_pdf_agent)
        if self.ocr_available:
            workflow.add_node("ocr_extraction", self._delegate_to_ocr_agent)
        workflow.add_node("llm_analysis", self._llm_analysis)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("finalization", self._finalize_with_llm)
        workflow.add_node("error_handler", self._handle_master_errors)
        
        # Define workflow
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "pdf_extraction")
        
        workflow.add_conditional_edges(
            "pdf_extraction",
            self._decide_extraction_method,
            {
                "analyze": "llm_analysis",
                **({"need_ocr": "ocr_extraction"} if self.ocr_available else {}),
                "error": "error_handler",
            }
        )
        
        if self.ocr_available:
            workflow.add_conditional_edges(
                "ocr_extraction",
                self._check_ocr_success,
                {
                    "analyze": "llm_analysis",
                    "error": "error_handler",
                }
            )
        
        # LLM analysis flow
        workflow.add_conditional_edges(
            "llm_analysis",
            self._route_after_analysis,
            {
                "use_tools": "tools",
                "finalize": "finalization",
                "error": "error_handler"
            }
        )
        
        workflow.add_edge("tools", "finalization")
        workflow.add_edge("finalization", "__end__")
        workflow.add_edge("error_handler", "__end__")
        
        return workflow.compile()
    
    def _initialize(self, state: MasterState) -> MasterState:
        """Initialize the extraction process"""
        logger.info("🚀 Enhanced Master Agent: Initializing with LLM support")
        
        pdf_path = state['pdf_path']
        if not pdf_path.endswith('.pdf'):
            return {
                **state,
                "error_messages": ["Invalid file format. Please provide a PDF file."],
                "current_step": "error",
                "messages": [HumanMessage(content=f"Error: Invalid file format for {pdf_path}")]
            }
        
        return {
            **state,
            "current_step": "initialized",
            "messages": [HumanMessage(content=f"Starting table extraction for: {pdf_path}")],
            "agent_recommendations": ["Initialized with LLM-powered analysis"]
        }
    
    def _delegate_to_pdf_agent(self, state: MasterState) -> MasterState:
        """Run PDF Agent extraction"""
        logger.info("📄 Enhanced Agent: Delegating to PDF Agent...")
        
        try:
            pdf_result = self.pdf_agent.extract_tables(state['pdf_path'])
            tables_found = pdf_result.get('tables_found', 0)
            confidence = pdf_result.get('overall_confidence', 0.0)
            
            logger.info(f"📊 PDF Results: {tables_found} tables, confidence: {confidence:.2f}")
            
            return {
                **state,
                "raw_extraction_result": pdf_result,
                "extraction_method": "pdfplumber",
                "current_step": "pdf_extraction_completed",
                "messages": [AIMessage(content=f"PDF extraction completed: {tables_found} tables found with {confidence:.2f} confidence")]
            }
            
        except Exception as e:
            error_msg = f"PDF Agent failed: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "error_messages": [error_msg],
                "current_step": "pdf_extraction_failed",
                "messages": [AIMessage(content=f"Error in PDF extraction: {error_msg}")]
            }
    
    def _delegate_to_ocr_agent(self, state: MasterState) -> MasterState:
        """Run OCR Agent extraction"""
        logger.info("👁️ Enhanced Agent: Delegating to OCR Agent...")
        
        try:
            ocr_result = self.ocr_agent.extract_tables_with_ocr(state['pdf_path'])
            tables_found = ocr_result.get('tables_found', 0)
            confidence = ocr_result.get('overall_confidence', 0.0)
            
            logger.info(f"👁️ OCR Results: {tables_found} tables, confidence: {confidence:.2f}")
            
            return {
                **state,
                "raw_extraction_result": ocr_result,
                "extraction_method": "ocr",
                "current_step": "ocr_extraction_completed",
                "messages": [AIMessage(content=f"OCR extraction completed: {tables_found} tables found with {confidence:.2f} confidence")]
            }
            
        except Exception as e:
            error_msg = f"OCR Agent failed: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "error_messages": [error_msg],
                "current_step": "ocr_extraction_failed",
                "messages": [AIMessage(content=f"Error in OCR extraction: {error_msg}")]
            }
    
    def _llm_analysis(self, state: MasterState) -> MasterState:
        """Use LLM to analyze extraction results and make decisions"""
        logger.info("🤖 Enhanced Agent: Running LLM analysis...")
        
        try:
            raw_result = state.get('raw_extraction_result', {})
            if not raw_result:
                return {
                    **state,
                    "error_messages": ["No extraction result to analyze"],
                    "current_step": "analysis_failed"
                }
            
            # Create analysis prompt
            tables_found = raw_result.get('tables_found', 0)
            confidence = raw_result.get('overall_confidence', 0.0)
            method = raw_result.get('extraction_method', 'unknown')
            
            analysis_prompt = f"""
            Analyze this table extraction result:
            - Method: {method}
            - Tables found: {tables_found}
            - Confidence: {confidence:.2f}
            - PDF: {state.get('pdf_path', 'unknown')}
            
            Provide analysis and next step recommendation.
            """
            
            if self.llm:
                try:
                    response = self.llm.invoke(analysis_prompt)
                    llm_analysis = response.strip() if isinstance(response, str) else str(response)
                except Exception as e:
                    logger.warning(f"LLM analysis failed: {e}")
                    llm_analysis = f"Rule-based analysis: Found {tables_found} tables with {confidence:.2f} confidence using {method}"
            else:
                llm_analysis = f"Rule-based analysis: Found {tables_found} tables with {confidence:.2f} confidence using {method}"
            
            # Make decision based on results
            if tables_found > 0 and confidence > 0.3:
                decision = "PROCEED: Acceptable extraction quality"
                next_step = "finalize"
            elif tables_found > 0:
                decision = "PROCEED_CAUTIOUSLY: Low confidence but tables found"
                next_step = "finalize"
            else:
                decision = "NO_TABLES: Consider alternative approach"
                next_step = "finalize"  # Still finalize with empty results
            
            return {
                **state,
                "llm_analysis": llm_analysis,
                "decision_reasoning": decision,
                "current_step": "analysis_completed",
                "messages": [
                    AIMessage(content=f"LLM Analysis: {llm_analysis}"),
                    AIMessage(content=f"Decision: {decision}")
                ]
            }
            
        except Exception as e:
            error_msg = f"LLM analysis failed: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "error_messages": [error_msg],
                "current_step": "analysis_failed",
                "llm_analysis": "Analysis failed - proceeding with default logic",
                "decision_reasoning": "Fallback to rule-based processing"
            }
    
    def _finalize_with_llm(self, state: MasterState) -> MasterState:
        """Enhanced finalization with LLM-generated summary"""
        logger.info("🎯 Enhanced Agent: Finalizing with LLM summary...")
        
        try:
            raw = state.get('raw_extraction_result') or {}
            extracted_values = self._collect_value_header_records(raw)
            
            # Generate LLM summary
            tables_found = len(raw.get('tables', []))
            data_points = len(extracted_values)
            
            if self.llm:
                try:
                    summary_prompt = f"""
                    Create a concise summary for this table extraction:
                    - Tables found: {tables_found}
                    - Data values extracted: {data_points}
                    - Method: {state.get('extraction_method', 'unknown')}
                    - Success: {'Yes' if tables_found > 0 else 'No'}
                    
                    Summary:"""
                    
                    llm_summary = self.llm.invoke(summary_prompt)
                    summary = llm_summary.strip() if isinstance(llm_summary, str) else str(llm_summary)
                except Exception:
                    summary = f"Extracted {data_points} data values from {tables_found} tables using {state.get('extraction_method', 'unknown')} method"
            else:
                summary = f"Extracted {data_points} data values from {tables_found} tables using {state.get('extraction_method', 'unknown')} method"
            
            final_output = {
                "master_agent_metadata": {
                    "extraction_method_used": state.get('extraction_method'),
                    "workflow_completed": True,
                    "llm_powered": self.llm is not None,
                    "total_agents_involved": 3 if self.llm else 2,
                    "final_confidence": raw.get("overall_confidence", 0.0),
                },
                "llm_analysis": {
                    "analysis": state.get('llm_analysis', 'Not available'),
                    "decision_reasoning": state.get('decision_reasoning', 'Rule-based'),
                    "summary": summary
                },
                "extracted_values": extracted_values,
                "agent_recommendations": state.get('agent_recommendations', []),
                "conversation_log": [
                    msg.content if hasattr(msg, 'content') else str(msg) 
                    for msg in state.get('messages', [])
                ]
            }
            
            return {
                **state,
                "final_json": final_output,
                "current_step": "completed"
            }
            
        except Exception as e:
            error_msg = f"Enhanced finalization failed: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                "error_messages": [error_msg],
                "current_step": "finalization_failed"
            }
    
    def _collect_value_header_records(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect value records with headers (same as original)"""
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
        """Enhanced error handling with LLM analysis"""
        error_messages = state.get('error_messages', [])
        logger.error(f"❌ Enhanced Agent: Handling errors: {error_messages}")
        
        return {
            **state,
            "current_step": "master_error",
            "final_json": {
                "master_agent_metadata": {
                    "extraction_method_used": state.get('extraction_method'),
                    "workflow_completed": False,
                    "error_occurred": True,
                    "llm_powered": self.llm is not None
                },
                "errors": error_messages,
                "final_state": state.get('current_step'),
                "llm_analysis": state.get('llm_analysis', 'Not available due to error')
            }
        }
    
    # Decision methods (enhanced with LLM awareness)
    def _decide_extraction_method(self, state: MasterState) -> str:
        """Enhanced decision making"""
        raw_result = state.get('raw_extraction_result', {}) or {}
        tables_found = raw_result.get('tables_found', 0)
        
        logger.info(f"🤔 Enhanced decision point: tables_found={tables_found}")
        
        if tables_found > 0:
            logger.info("✅ PDF extraction has tables → analyzing with LLM")
            return "analyze"
        elif self.ocr_available:
            logger.info("📄 No tables found via PDF → trying OCR...")
            return "need_ocr"
        else:
            logger.info("❌ No tables found and OCR not available")
            return "analyze"  # Still analyze the empty result
    
    def _check_ocr_success(self, state: MasterState) -> str:
        """Enhanced OCR success check"""
        raw_result = state.get('raw_extraction_result', {}) or {}
        tables_found = raw_result.get('tables_found', 0)
        logger.info(f"🤔 Enhanced OCR decision: tables_found={tables_found}")
        return "analyze"  # Always analyze OCR results with LLM
    
    def _route_after_analysis(self, state: MasterState) -> str:
        """Route after LLM analysis"""
        # For now, go directly to finalization
        # Could be enhanced to use tools based on LLM decision
        return "finalize"
    
    # Public API (same interface)
    def extract_tables(self, pdf_path: str) -> Dict[str, Any]:
        """Run the enhanced LLM-powered workflow"""
        logger.info(f"🚀 Enhanced Master Agent: Starting LLM-powered extraction for: {pdf_path}")
        
        initial_state: MasterState = {
            "pdf_path": pdf_path,
            "extraction_method": None,
            "raw_extraction_result": None,
            "final_json": None,
            "error_messages": [],
            "current_step": "master_initialized",
            "agent_recommendations": [],
            "messages": [],
            "llm_analysis": None,
            "decision_reasoning": None
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            self._print_execution_summary(final_state)
            return final_state.get('final_json', {
                "error": "Enhanced workflow failed",
                "error_messages": final_state.get('error_messages', []),
                "final_state": final_state.get('current_step')
            })
            
        except Exception as e:
            logger.error(f"Enhanced workflow execution failed: {e}")
            return {
                "error": f"Enhanced workflow failed: {str(e)}",
                "error_messages": [str(e)],
                "final_state": "enhanced_workflow_error"
            }
    
    def _print_execution_summary(self, final_state: MasterState):
        """Enhanced execution summary"""
        print(f"\n{'='*60}")
        print("🤖 ENHANCED MASTER AGENT EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        method = final_state.get('extraction_method', 'unknown')
        current_step = final_state.get('current_step', 'unknown')
        
        print(f"📄 PDF File: {final_state.get('pdf_path', 'unknown')}")
        print(f"🔧 Extraction Method: {method}")
        print(f"📊 Final Step: {current_step}")
        print(f"🤖 LLM Powered: {'Yes' if self.llm else 'No (fallback mode)'}")
        
        # Show LLM analysis
        llm_analysis = final_state.get('llm_analysis')
        if llm_analysis:
            print(f"\n🧠 LLM Analysis:")
            print(f"   {llm_analysis}")
        
        decision = final_state.get('decision_reasoning')
        if decision:
            print(f"\n🎯 Decision Reasoning:")
            print(f"   {decision}")
        
        # Show errors if any
        errors = final_state.get('error_messages', [])
        if errors:
            print(f"\n❌ Errors Encountered: {len(errors)}")
            for i, error in enumerate(errors[-3:]):
                print(f"   {i+1}. {error}")
        
        # Show extracted values count
        fj = final_state.get("final_json") or {}
        values = fj.get("extracted_values", []) if isinstance(fj, dict) else []
        print(f"\n✅ Extracted value records: {len(values)}")
        print(f"{'='*60}\n")


def main():
    """Run the enhanced extraction with LLM support"""
    print("🚀 Enhanced Table Extraction with Hugging Face LLM")
    print("="*50)
    
    pdf_path = input("📁 Enter the path to your PDF file: ").strip()
    if not pdf_path:
        pdf_path = "data/example.pdf"
        print(f"Using default path: {pdf_path}")
    
    try:
        # Initialize enhanced master agent with LLM
        master = EnhancedMasterAgent(model_name="microsoft/DialoGPT-small")
        
        # Run extraction
        results = master.extract_tables(pdf_path)
        
        # Save results
        output_file = "results/enhanced_extraction_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_file}")
        
        # Print final JSON (formatted)
        print("\n📋 FINAL JSON OUTPUT (LLM-Enhanced):")
        print("=" * 50)
        print(json.dumps(results, indent=2))
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced execution failed: {e}")
        print(f"❌ Fatal Error: {e}")
        return {"fatal_error": str(e)}


if __name__ == "__main__":
    results = main()