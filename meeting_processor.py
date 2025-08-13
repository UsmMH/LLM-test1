"""
Meeting Minutes Processor
Handles all LLM operations and workflow logic for meeting transcript analysis.
"""

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define state schema for better type safety
class MeetingState(TypedDict):
    transcript: str
    summary: str
    action_items: str
    decisions: str

class MeetingProcessor:
    """Handles meeting transcript processing using LLM and LangGraph workflow."""
    
    def __init__(self, model_name: str = "qwen3:4b", timeout: int = 60):
        """
        Initialize the meeting processor with LLM and workflow.
        
        Args:
            model_name: Name of the Ollama model to use
            timeout: Timeout for LLM requests in seconds
        """
        self.model_name = model_name
        self.timeout = timeout
        self.llm = None
        self.workflow = None
        self._initialize_llm()
        self._setup_prompts()
        self._create_workflow()
    
    def _initialize_llm(self) -> None:
        """Initialize the Ollama model with error handling."""
        try:
            # Initialize with cleaner parameters to reduce warnings
            self.llm = Ollama(
                model=self.model_name, 
                timeout=self.timeout,
                # Reduce unnecessary parameters that cause warnings
                mirostat=None,
                mirostat_eta=None,
                mirostat_tau=None,
                tfs_z=None
            )
            # Test the model with a simple query
            test_response = self.llm.invoke("Test connection")
            logger.info(f"LLM initialized successfully with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise Exception(f"Failed to load the AI model: {e}. Please ensure Ollama is running and the {self.model_name} model is installed.")
    
    def _setup_prompts(self) -> None:
        """Setup prompt templates for different processing tasks."""
        self.summary_prompt = PromptTemplate.from_template(
            """Summarize the following meeting transcript in a single, well-structured paragraph. 
            Focus on the main topics discussed, key points raised, and overall outcomes.
            Provide ONLY the summary paragraph - no additional text, explanations, or thinking process.
            
            Meeting Transcript:
            {transcript}
            
            Summary:"""
        )
        
        self.action_items_prompt = PromptTemplate.from_template(
            """Extract all action items from the following meeting transcript. 
            For each item, identify:
            1. The specific task or action to be taken
            2. The person responsible (if mentioned)
            3. Any deadlines or timeframes (if mentioned)
            
            If no person is specified, write 'Unassigned'.
            Format the output as a bulleted list using markdown format.
            Provide ONLY the action items list - no additional text, explanations, or thinking process.
            
            Meeting Transcript:
            {transcript}
            
            Action Items:"""
        )
        
        self.decisions_prompt = PromptTemplate.from_template(
            """Identify and list all key decisions made in the following meeting transcript.
            Include both explicit decisions and implicit agreements or conclusions reached.
            Format the output as a bulleted list using markdown format.
            Provide ONLY the decisions list - no additional text, explanations, or thinking process.
            
            Meeting Transcript:
            {transcript}
            
            Key Decisions:"""
        )
    
    def _generate_summary(self, state: MeetingState) -> Dict[str, Any]:
        """Generate a summary of the meeting transcript."""
        try:
            transcript = state["transcript"]
            if not transcript.strip():
                return {"summary": "No transcript provided for summary generation."}
            
            prompt = self.summary_prompt.format(transcript=transcript)
            summary = self.llm.invoke(prompt)
            logger.info("Summary generated successfully")
            
            # Clean up any unwanted content like thinking tags
            cleaned_summary = self._clean_response(summary)
            return {"summary": cleaned_summary}
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"summary": f"Error generating summary: {str(e)}"}
    
    def _extract_action_items(self, state: MeetingState) -> Dict[str, Any]:
        """Extract action items from the meeting transcript."""
        try:
            transcript = state["transcript"]
            if not transcript.strip():
                return {"action_items": "No transcript provided for action item extraction."}
            
            prompt = self.action_items_prompt.format(transcript=transcript)
            action_items = self.llm.invoke(prompt)
            logger.info("Action items extracted successfully")
            
            # Clean up any unwanted content
            cleaned_action_items = self._clean_response(action_items)
            return {"action_items": cleaned_action_items}
        except Exception as e:
            logger.error(f"Error extracting action items: {e}")
            return {"action_items": f"Error extracting action items: {str(e)}"}
    
    def _extract_decisions(self, state: MeetingState) -> Dict[str, Any]:
        """Extract key decisions from the meeting transcript."""
        try:
            transcript = state["transcript"]
            if not transcript.strip():
                return {"decisions": "No transcript provided for decision extraction."}
            
            prompt = self.decisions_prompt.format(transcript=transcript)
            decisions = self.llm.invoke(prompt)
            logger.info("Decisions extracted successfully")
            
            # Clean up any unwanted content
            cleaned_decisions = self._clean_response(decisions)
            return {"decisions": cleaned_decisions}
        except Exception as e:
            logger.error(f"Error extracting decisions: {e}")
            return {"decisions": f"Error extracting decisions: {str(e)}"}
    
    def _create_workflow(self) -> None:
        """Create and compile the LangGraph workflow."""
        workflow = StateGraph(MeetingState)
        
        # Add the nodes
        workflow.add_node("summary", self._generate_summary)
        workflow.add_node("action_items", self._extract_action_items)
        workflow.add_node("decisions", self._extract_decisions)
        
        # Set entry point
        workflow.set_entry_point("summary")
        
        # Run tasks in parallel for better performance
        workflow.add_edge("summary", "action_items")
        workflow.add_edge("summary", "decisions")
        
        # Both action_items and decisions end the workflow
        workflow.add_edge("action_items", "__end__")
        workflow.add_edge("decisions", "__end__")
        
        self.workflow = workflow.compile()
        logger.info("Workflow created and compiled successfully")
    
    def _clean_response(self, response: str) -> str:
        """
        Clean up LLM response by removing unwanted content like thinking tags.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned response text
        """
        import re
        
        # Remove <think>...</think> tags and their content
        cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove any other common unwanted patterns
        cleaned = re.sub(r'<.*?>', '', cleaned, flags=re.DOTALL)
        
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Multiple newlines to double
        cleaned = cleaned.strip()
        
        return cleaned
    
    def process_transcript(self, transcript: str) -> Dict[str, str]:
        """
        Process a meeting transcript and return structured results.
        
        Args:
            transcript: The meeting transcript text to process
            
        Returns:
            Dictionary containing summary, action_items, and decisions
            
        Raises:
            Exception: If processing fails
        """
        try:
            if not transcript or not transcript.strip():
                raise ValueError("Empty transcript provided")
            
            # Run the workflow
            result = self.workflow.invoke({"transcript": transcript})
            
            return {
                "summary": result.get("summary", "No summary could be generated."),
                "action_items": result.get("action_items", "No action items could be found."),
                "decisions": result.get("decisions", "No decisions could be found.")
            }
        except Exception as e:
            logger.error(f"Error processing transcript: {e}")
            raise Exception(f"Failed to process transcript: {str(e)}")
    
    def validate_transcript(self, transcript: str) -> tuple[bool, str]:
        """
        Validate the input transcript.
        
        Args:
            transcript: The transcript text to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not transcript:
            return False, "Please provide a transcript to process."
        
        if len(transcript.strip()) < 20:
            return False, "Please provide a more substantial transcript (at least 20 characters)."
        
        if len(transcript.strip()) > 50000:
            return False, "Transcript is too long. Please limit to 50,000 characters."
        
        return True, ""
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model configuration."""
        return {
            "model_name": self.model_name,
            "timeout": str(self.timeout),
            "status": "Ready" if self.llm and self.workflow else "Not Ready"
        }

# Factory function for easy initialization
def create_meeting_processor(model_name: str = "qwen3:4b", timeout: int = 60) -> MeetingProcessor:
    """
    Factory function to create a MeetingProcessor instance.
    
    Args:
        model_name: Name of the Ollama model to use
        timeout: Timeout for LLM requests in seconds
        
    Returns:
        Configured MeetingProcessor instance
    """
    return MeetingProcessor(model_name=model_name, timeout=timeout)