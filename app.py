import os
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import time

load_dotenv()

@dataclass
class SafetyResponse:
    """Structure for storing comprehensive safety responses"""
    category: str
    immediate_actions: List[str]
    detailed_steps: List[str]
    precautions: List[str]
    emergency_contacts: Dict[str, str]
    followup_care: List[str]
    additional_resources: Dict[str, str]
    confidence_score: float
    response_time: float

class IndianEmergencySystem:
    """Enhanced emergency response system for Indian context"""
    
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="mixtral-8x7b-32768",  # Using a more capable model
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.chat_history = []
        self.emergency_contacts = {
            "Police": "100",
            "Ambulance": "108",
            "Women Helpline": "1091",
            "Child Helpline": "1098",
            "National Emergency": "112",
            "Fire": "101",
            "Disaster Management": "1070"
        }
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize specialized emergency response agents"""
        self.agents = {
            "medical": self._create_medical_agent(),
            "personal_safety": self._create_safety_agent(),
            "disaster": self._create_disaster_agent(),
            "women_safety": self._create_women_safety_agent()
        }

    def _create_medical_agent(self):
        prompt = """You are an Indian Emergency Medical Response Expert.
Current Situation: {query}
Previous Context: {chat_history}

Provide a detailed medical emergency response following this exact structure:

🚨 IMMEDIATE ACTIONS (First 5 minutes):
- List specific life-saving steps
- Include local emergency numbers (108/112)
- Nearest hospital guidance

🏥 DETAILED MEDICAL STEPS:
- Step-by-step first aid instructions
- Common mistakes to avoid
- Local alternatives if modern equipment unavailable

⚕️ MEDICAL PRECAUTIONS:
- What NOT to do
- Warning signs to watch
- When to call for emergency help

📞 EMERGENCY CONTACTS:
- Local ambulance services
- Poison control centers
- Blood banks

🔄 FOLLOW-UP CARE:
- Recovery monitoring steps
- Required medical documentation
- Prevention measures

Ensure all guidance is practical for Indian conditions and resources.
Include specific instructions for both urban and rural settings.
Focus on immediate life-saving actions first."""

        return self._create_agent(prompt)

    def _create_safety_agent(self):
        prompt = """You are an Indian Personal Safety Expert.
Current Situation: {query}
Previous Context: {chat_history}

Provide comprehensive safety guidance following this structure:

🚨 IMMEDIATE SAFETY ACTIONS:
- Escape/protection steps
- Emergency numbers (100/112)
- Local police contact

🛡️ DETAILED SAFETY STEPS:
- Step-by-step protection measures
- De-escalation techniques
- Self-defense basics

⚠️ SAFETY PRECAUTIONS:
- Situation assessment
- Risk mitigation
- Prevention strategies

📞 EMERGENCY CONTACTS:
- Local police stations
- Community helplines
- Support organizations

🔄 FOLLOW-UP MEASURES:
- Documentation needed
- Legal options
- Support resources

Include specific Indian laws and regulations.
Consider both urban and rural contexts.
Focus on practical, implementable steps."""

        return self._create_agent(prompt)

    def _create_disaster_agent(self):
        prompt = """You are an Indian Disaster Management Expert.
Current Situation: {query}
Previous Context: {chat_history}

Provide detailed disaster response guidance:

🚨 IMMEDIATE ACTIONS:
- Evacuation steps
- Safety zones
- Emergency kit items

🏘️ DETAILED RESPONSE STEPS:
- Location-specific measures
- Resource management
- Group coordination

⚠️ SAFETY PRECAUTIONS:
- Area assessment
- Risk identification
- Preventive measures

📞 EMERGENCY CONTACTS:
- Disaster helpline (1070)
- Relief organizations
- Local authorities

🔄 RECOVERY GUIDANCE:
- Post-disaster assessment
- Relief camp locations
- Government assistance

Consider monsoon, earthquake, and flood scenarios.
Include both urban and rural contexts.
Focus on Indian disaster response protocols."""

        return self._create_agent(prompt)

    def _create_women_safety_agent(self):
        prompt = """You are an Indian Women's Safety Expert.
Current Situation: {query}
Previous Context: {chat_history}

Provide women-specific safety guidance:

🚨 IMMEDIATE ACTIONS:
- Safety steps
- Emergency numbers (1091/112)
- Safe escape tactics

👩 DETAILED SAFETY STEPS:
- Protection measures
- Documentation needs
- Support resources

⚠️ SAFETY PRECAUTIONS:
- Situation assessment
- Risk mitigation
- Prevention strategies

📞 EMERGENCY CONTACTS:
- Women helpline
- NGO support
- Legal aid

🔄 FOLLOW-UP MEASURES:
- Legal documentation
- Support groups
- Counseling services

Include specific Indian women protection laws.
Reference local women support organizations.
Focus on practical, immediate safety."""

        return self._create_agent(prompt)

    def _create_agent(self, prompt):
        return ChatPromptTemplate.from_messages([
            ("system", prompt),
            ("human", "{input}")
        ]) | self.llm | StrOutputParser()

    async def get_emergency_response(self, query: str) -> SafetyResponse:
        """Process emergency query and return comprehensive response"""
        start_time = time.time()
        
        # Determine emergency type and get relevant responses
        responses = await self._get_agent_responses(query)
        
        # Synthesize final response
        final_response = await self._create_final_response(responses, query)
        
        processing_time = time.time() - start_time
        
        return SafetyResponse(
            category=self._determine_emergency_type(query),
            immediate_actions=final_response["immediate_actions"],
            detailed_steps=final_response["detailed_steps"],
            precautions=final_response["precautions"],
            emergency_contacts=self.emergency_contacts,
            followup_care=final_response["followup"],
            additional_resources=final_response["resources"],
            confidence_score=final_response["confidence"],
            response_time=processing_time
        )

    async def _get_agent_responses(self, query: str) -> Dict[str, str]:
        """Get responses from all relevant agents"""
        tasks = []
        for agent_name, agent in self.agents.items():
            tasks.append(self._get_single_agent_response(agent_name, agent, query))
        
        responses = await asyncio.gather(*tasks)
        return dict(responses)

    async def _get_single_agent_response(self, name: str, agent, query: str):
        """Get response from a single agent"""
        try:
            response = await agent.ainvoke({
                "input": query,
                "query": query,
                "chat_history": self._format_chat_history()
            })
            return (name, response)
        except Exception as e:
            return (name, f"Error in {name}: {str(e)}")

    def _determine_emergency_type(self, query: str) -> str:
        """Determine the primary type of emergency from the query"""
        # Add logic to categorize emergency type
        return "general"

    def _format_chat_history(self) -> str:
        """Format recent chat history"""
        history = []
        for msg in self.chat_history[-5:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history.append(f"{role}: {msg.content}")
        return "\n".join(history)

    async def _create_final_response(self, responses: Dict[str, str], query: str) -> Dict:
        """Create final synthesized response"""
        synthesis_prompt = """
        Analyze these emergency responses and create a comprehensive safety guide:

        Original Query: {query}

        Agent Responses:
        {responses}

        Create a structured response with:
        1. Most critical immediate actions
        2. Detailed step-by-step guidance
        3. Important precautions
        4. Follow-up care steps
        5. Additional resources and contacts

        Focus on practical, actionable steps for Indian context.
        """

        try:
            synthesis_chain = ChatPromptTemplate.from_messages([
                ("system", synthesis_prompt)
            ]) | self.llm | StrOutputParser()

            synthesis = await synthesis_chain.ainvoke({
                "query": query,
                "responses": "\n\n".join([f"{k}:\n{v}" for k, v in responses.items()])
            })

            # Parse synthesis into structured format
            return {
                "immediate_actions": self._extract_section(synthesis, "IMMEDIATE ACTIONS"),
                "detailed_steps": self._extract_section(synthesis, "DETAILED STEPS"),
                "precautions": self._extract_section(synthesis, "PRECAUTIONS"),
                "followup": self._extract_section(synthesis, "FOLLOW-UP"),
                "resources": self._extract_resources(synthesis),
                "confidence": 0.9  # Add confidence scoring logic
            }
        except Exception as e:
            raise Exception(f"Response synthesis error: {str(e)}")

    def _extract_section(self, text: str, section: str) -> List[str]:
        """Extract specific section from synthesis response"""
        # Add logic to extract and parse sections
        return ["Sample step 1", "Sample step 2"]

    def _extract_resources(self, text: str) -> Dict[str, str]:
        """Extract additional resources from synthesis response"""
        # Add logic to extract resources
        return {"Resource 1": "Contact info"}

def setup_streamlit_ui():
    """Configure Streamlit UI with enhanced styling"""
    st.set_page_config(
        page_title="Indian Emergency Response System",
        page_icon="🚨",
        layout="wide"
    )

    # Add custom CSS for better UI
    st.markdown("""
        <style>
        .emergency-header {
            background-color: #DC3545;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .response-card {
            background-color: #F8F9FA;
            border-left: 5px solid #28A745;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 1rem;
        }
        
        .emergency-contact {
            background-color: #FFC107;
            color: black;
            padding: 0.5rem;
            border-radius: 5px;
            margin: 0.5rem 0;
            display: inline-block;
        }
        
        .step-card {
            background-color: #E9ECEF;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    setup_streamlit_ui()
    
    # Initialize session state
    if "emergency_system" not in st.session_state:
        st.session_state.emergency_system = IndianEmergencySystem()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Application header
    st.markdown("""
        <div class="emergency-header">
            <h1>🚨 Indian Emergency Response System</h1>
            <p>Get immediate guidance for emergency situations</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar with emergency contacts
    with st.sidebar:
        st.markdown("### 📞 Emergency Contacts")
        for service, number in st.session_state.emergency_system.emergency_contacts.items():
            st.markdown(f"""
                <div class="emergency-contact">
                    {service}: {number}
                </div>
            """, unsafe_allow_html=True)

    # Main chat interface
    if query := st.chat_input("Describe your emergency situation..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        with st.spinner("Analyzing emergency situation..."):
            try:
                response = asyncio.run(
                    st.session_state.emergency_system.get_emergency_response(query)
                )
                
                # Display response
                st.markdown("""
                    <div class="response-card">
                        <h3>🚨 Immediate Actions Required:</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                for action in response.immediate_actions:
                    st.markdown(f"""
                        <div class="step-card">
                            ✓ {action}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Display other sections
                sections = [
                    ("📋 Detailed Steps", response.detailed_steps),
                    ("⚠️ Important Precautions", response.precautions),
                    ("🔄 Follow-up Care", response.followup_care)
                ]
                
                for title, steps in sections:
                    with st.expander(title):
                        for step in steps:
                            st.markdown(f"- {step}")
                
                # Update chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": str(response)
                })
                
            except Exception as e:
                st.error(f"Error processing emergency response: {str(e)}")

if __name__ == "__main__":
    main()
