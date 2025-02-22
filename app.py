import os
import asyncio
from dataclasses import dataclass
from typing import Dict, List
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from dotenv import load_dotenv
import time

load_dotenv()

@dataclass
class AgentResponse:
    """Structure for storing safety agent responses"""
    agent_name: str
    content: str
    confidence: float
    metadata: Dict = None
    processing_time: float = 0.0

class AgentStatus:
    def __init__(self):
        self.sidebar_placeholder = None
        self.agents = {
            'medical_response': {'status': 'idle', 'progress': 0, 'message': ''},
            'personal_safety': {'status': 'idle', 'progress': 0, 'message': ''},
            'disaster_response': {'status': 'idle', 'progress': 0, 'message': ''},
            'women_safety': {'status': 'idle', 'progress': 0, 'message': ''}
        }
        
    def initialize_sidebar_placeholder(self):
        with st.sidebar:
            self.sidebar_placeholder = st.empty()
    
    def update_status(self, agent_name: str, status: str, progress: float, message: str = ""):
        self.agents[agent_name] = {
            'status': status,
            'progress': progress,
            'message': message
        }
        self._render_status()

    def _render_status(self):
        if self.sidebar_placeholder is None:
            self.initialize_sidebar_placeholder()
            
        with self.sidebar_placeholder.container():
            for agent_name, status in self.agents.items():
                self._render_agent_card(agent_name, status)

    def _render_agent_card(self, agent_name: str, status: dict):
        colors = {
            'idle': '#6c757d',
            'working': '#ffd700',  # Yellow while working
            'completed': '#28a745',  # Green when completed
            'error': '#dc3545'
        }
        color = colors.get(status['status'], colors['idle'])
        
        animation_css = """
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        """
        
        animation_style = "animation: pulse 2s infinite;" if status['status'] == 'working' else ""
        
        st.markdown(f"""
            <style>{animation_css}</style>
            <div style="
                background-color: #1E1E1E;
                padding: 0.8rem;
                border-radius: 0.5rem;
                margin-bottom: 0.8rem;
                border: 1px solid {color};
                {animation_style}
                transition: all 0.3s ease;
            ">
                <div style="color: {color}; font-weight: bold;">
                    {agent_name.replace('_', ' ').title()}
                </div>
                <div style="
                    color: #FFFFFF;
                    font-size: 0.8rem;
                    margin: 0.3rem 0;
                ">
                    {status['message'] or status['status'].title()}
                </div>
                <div style="
                    height: 4px;
                    background-color: rgba(255,255,255,0.1);
                    border-radius: 2px;
                    margin-top: 0.5rem;
                ">
                    <div style="
                        width: {status['progress'] * 100}%;
                        height: 100%;
                        background-color: {color};
                        border-radius: 2px;
                        transition: width 0.3s ease;
                    "></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

class SafetyResponseSystem:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.3,
            model_name="llama-guard-3-8b",
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.chat_history = []
        self._initialize_prompts()
        self.agents = self._initialize_agents()

    def _initialize_prompts(self):
        self.prompts = {
            'medical_response': """You are an Indian Emergency Medical Response AI assistant.
Query: {query}
Chat History: {chat_history}

Provide detailed step-by-step medical guidance for India:

1. IMMEDIATE ACTIONS:
   - Specific first-aid steps
   - Critical do's and don'ts
   
2. EMERGENCY CONTACTS:
   - When to call 102 (Ambulance)
   - When to call 108 (Emergency)
   - Local hospital contact information
   
3. MEDICAL ASSISTANCE:
   - Signs requiring immediate professional help
   - Important medical information to share
   
4. FOLLOW-UP CARE:
   - Post-emergency care steps
   - Medical documentation needed
   
Focus on practical, life-saving steps appropriate for Indian healthcare context.""",

            'personal_safety': """You are an Indian Personal Safety Response AI assistant.
Query: {query}
Chat History: {chat_history}

Provide specific safety guidance for Indian context:

1. IMMEDIATE SAFETY:
   - Location-specific safety steps
   - Contact emergency number 112
   - Local police station contact
   
2. PROTECTION MEASURES:
   - Practical defense strategies
   - Safe zones identification
   - Community support access
   
3. EMERGENCY RESOURCES:
   - Local authorities contact
   - Support organizations
   - Documentation needed
   
4. PREVENTIVE STEPS:
   - Future safety measures
   - Alert systems setup
   - Emergency planning

Keep instructions practical and specific to Indian environment.""",

            'disaster_response': """You are an Indian Disaster Response AI assistant.
Query: {query}
Chat History: {chat_history}

Provide India-specific disaster response guidance:

1. IMMEDIATE ACTIONS:
   - Location-based safety steps
   - Evacuation procedures
   - Emergency kit essentials
   
2. EMERGENCY CONTACTS:
   - NDRF helpline (1078)
   - State disaster management
   - Local emergency services
   
3. DISASTER PROTOCOLS:
   - Area-specific guidelines
   - Communication methods
   - Resource management
   
4. RECOVERY STEPS:
   - Post-disaster safety
   - Government assistance
   - Documentation needed

Focus on practical steps relevant to Indian disaster response systems.""",

            'women_safety': """You are an Indian Women's Safety Response AI assistant.
Query: {query}
Chat History: {chat_history}

Provide focused safety guidance for Indian women:

1. IMMEDIATE SAFETY:
   - Women's helpline (1091)
   - Police helpline (112)
   - Immediate protection steps
   
2. SUPPORT RESOURCES:
   - Local women's organizations
   - Legal aid services
   - Safe shelter locations
   
3. LEGAL RIGHTS:
   - Indian legal protections
   - Documentation needed
   - Reporting procedures
   
4. SAFETY STRATEGIES:
   - Practical protection methods
   - Emergency alert setup
   - Support network building

Maintain sensitivity while providing practical, India-specific guidance."""
        }

    def _initialize_agents(self):
        return {
            name: ChatPromptTemplate.from_messages([
                ("system", prompt),
                ("human", "{input}")
            ]) | self.llm | StrOutputParser()
            for name, prompt in self.prompts.items()
        }

    def _format_chat_history(self) -> str:
        formatted = []
        for msg in self.chat_history[-5:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            formatted.append(f"{role}: {msg.content}")
        return "\n".join(formatted)

    async def process_query(
        self,
        query: str,
        status_callback
    ) -> Dict[str, AgentResponse]:
        responses = {}
        chat_history = self._format_chat_history()
        
        try:
            agent_tasks = []
            for agent_name in self.agents.keys():
                status_callback(agent_name, 'working', 0.2, f"Analyzing situation")
                agent_tasks.append(self._get_agent_response(
                    agent_name, query, chat_history
                ))

            agent_responses = await asyncio.gather(*agent_tasks)
            
            for agent_name, response in zip(self.agents.keys(), agent_responses):
                responses[agent_name] = response
                status_callback(
                    agent_name,
                    'completed',
                    1.0,
                    f"Analysis complete"
                )

            final_response = await self._synthesize_safety_responses(
                query, chat_history, responses
            )
            responses['final_analysis'] = final_response

            self.chat_history.extend([
                HumanMessage(content=query),
                AIMessage(content=final_response.content)
            ])

            return responses

        except Exception as e:
            for agent in self.agents.keys():
                status_callback(agent, 'error', 0, str(e))
            raise Exception(f"Analysis error: {str(e)}")

    async def _get_agent_response(
        self,
        agent_name: str,
        query: str,
        chat_history: str
    ) -> AgentResponse:
        start_time = time.time()
        
        try:
            response = await self.agents[agent_name].ainvoke({
                "input": query,
                "query": query,
                "chat_history": chat_history
            })
            
            processing_time = time.time() - start_time
            
            metadata = {
                "processing_time": processing_time,
                "query_length": len(query),
                "response_confidence": self._calculate_confidence(response)
            }
            
            return AgentResponse(
                agent_name=agent_name,
                content=response,
                confidence=self._calculate_confidence(response),
                metadata=metadata,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise Exception(f"Agent {agent_name} error: {str(e)}")

    def _calculate_confidence(self, response: str) -> float:
        confidence = 0.7
        
        key_indicators = [
            "immediately", "urgent", "emergency",
            "call 112", "call 108", "call 102",
            "seek help", "safety", "caution", "warning"
        ]
        
        for indicator in key_indicators:
            if indicator.lower() in response.lower():
                confidence += 0.03
                
        return min(0.95, confidence)

    async def _synthesize_safety_responses(
        self,
        query: str,
        chat_history: str,
        responses: Dict[str, AgentResponse]
    ) -> AgentResponse:
        try:
            greetings = ['hi', 'hello', 'hey', 'hii', 'greetings']
            if query.lower().strip() in greetings:
                return AgentResponse(
                    agent_name="final_analysis",
                    content="Namaste! I'm your Indian Emergency Response Assistant. I can help you with medical emergencies, personal safety, disaster response, and women's safety. How may I assist you?",
                    confidence=1.0,
                    metadata={"greeting": True},
                    processing_time=0.1
                )

            formatted_responses = "\n\n".join([
                f"{name.upper()}:\n{response.content}"
                for name, response in responses.items()
                if name != 'final_analysis'
            ])

            start_time = time.time()
            
            synthesis_template = """
            Analyze these safety recommendations and provide India-specific guidance:

            1. IMMEDIATE ACTIONS:
               - Critical first steps
               - Emergency numbers to call
               - Local authority contact
            
            2. DETAILED STEPS:
               - Step-by-step instructions
               - Important precautions
               - Resource requirements
            
            3. PROFESSIONAL HELP:
               - When to seek emergency services
               - Required documentation
               - Contact information
            
            4. FOLLOW-UP ACTIONS:
               - Post-emergency steps
               - Documentation needs
               - Support resources

            Safety Assessments:
            {formatted_responses}
            
            Provide clear, actionable steps focused on Indian emergency response systems.
            Always include relevant emergency numbers (112, 108, 102, 1091, 1078).
            """

            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", synthesis_template)
            ])
            
            synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
            
            synthesis_response = await synthesis_chain.ainvoke({
                "formatted_responses": formatted_responses
            })
            
            processing_time = time.time() - start_time
            
            overall_confidence = sum(
                response.confidence for response in responses.values()
            ) / len(responses)
            
            metadata = {
                "processing_time": processing_time,
                "source_responses": len(responses),
                "overall_confidence": overall_confidence
            }
            
            return AgentResponse(
                agent_name="final_analysis",
                content=synthesis_response,
                confidence=overall_confidence,
                metadata=metadata,
                processing_time=processing_time
            )

        except Exception as e:
            raise Exception(f"Synthesis error: {str(e)}")

def setup_streamlit_ui():
    st.set_page_config(
        page_title="Indian Emergency Response System",
        page_icon="🚨",
        layout="wide"
    )
    
    st.markdown("""
        <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }
        
        .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            border: 1px solid #28a745;
            background-color: rgba(40, 167, 69, 0.1);
            animation: fadeIn 0.5s ease-out;
        }
        
        .agent-card {
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #28a745;
            border-radius: 0.5rem;
            background-color: rgba(40, 167, 69, 0.05);
            animation: fadeIn 0.5s ease-out;
        }
        
        .metadata-section {
            font-size: 0.8rem;
            color: #28a745;
            margin-top: 0.5rem;
        }
        
        .emergency-button {
            background-color: #dc3545;
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            cursor: pointer;
            margin: 1rem 0;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .emergency-button:hover {
            background-color: #c82333;
            transform: scale(1.02);
        }
        
        [data-testid="stSidebar"] {
            background-color: #1a1a1a;
        }
        
        .typing-animation {
            overflow: hidden;
            white-space: nowrap;
            border-right: 2px solid #28a745;
            animation: typing 3.5s steps(40, end),
                       blink-caret 0.75s step-end infinite;
        }
        
        @keyframes blink-caret {
            from, to { border-color: transparent }
            50% { border-color: #28a745 }
        }
        
        .emergency-numbers {
            background-color: rgba(220, 53, 69, 0.1);
            border: 1px solid #dc3545;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        
        .number-badge {
            background-color: #28a745;
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 0.25rem;
            margin-right: 0.5rem;
            font-weight: bold;
        }
        
        .expander-content {
            background-color: rgba(40, 167, 69, 0.05);
            border-radius: 0.5rem;
            padding: 1rem;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    setup_streamlit_ui()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        st.session_state.agent = SafetyResponseSystem()
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = AgentStatus()
    
    st.markdown("""
        <div style='text-align: center; color: #ffffff;'>
            <h1 class='typing-animation'>🚨 Indian Emergency Response System</h1>
            <p>Get immediate guidance for emergency situations in India</p>
        </div>
        
        <div class="emergency-numbers">
            <h3>Important Emergency Numbers</h3>
            <p><span class="number-badge">112</span> National Emergency Number</p>
            <p><span class="number-badge">102</span> Ambulance Services</p>
            <p><span class="number-badge">108</span> Emergency Medical Services</p>
            <p><span class="number-badge">1091</span> Women's Helpline</p>
            <p><span class="number-badge">1078</span> Disaster Management</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with emergency resources
    with st.sidebar:
        st.markdown("""
            <div style='color: #28a745;'>
                <h3>🏥 Emergency Resources</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class='emergency-button'>
                <span style='font-size: 1.2rem;'>📞</span>
                Emergency: 112 (All India)
            </div>
            
            <div style='margin-top: 2rem;'>
                <h4>Quick Access Situations:</h4>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🏥 Medical Emergency", key="medical"):
            st.session_state.messages.append({
                "role": "user",
                "content": "I need immediate medical emergency assistance. What should I do? Please provide step-by-step guidance."
            })
        
        if st.button("🛡️ Personal Safety", key="safety"):
            st.session_state.messages.append({
                "role": "user",
                "content": "I'm in an unsafe situation and need immediate help. What steps should I take for my safety?"
            })
            
        if st.button("🌪️ Natural Disaster", key="disaster"):
            st.session_state.messages.append({
                "role": "user",
                "content": "There's a natural disaster in my area. What immediate steps should I take for safety?"
            })
            
        if st.button("👩 Women's Safety", key="women"):
            st.session_state.messages.append({
                "role": "user",
                "content": "I need urgent assistance regarding women's safety. What immediate steps should I take?"
            })
        
        st.markdown("""
            <div style='color: #28a745; margin-top: 2rem;'>
                <h3>🤖 Response Agents Status</h3>
            </div>
        """, unsafe_allow_html=True)
        st.session_state.agent_status.initialize_sidebar_placeholder()
    
    # Chat interface with animations
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {message['content']['final_analysis'].content}
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔍 Detailed Analysis Report", expanded=False):
                    st.markdown('<div class="expander-content">', unsafe_allow_html=True)
                    for agent_name, response in message['content'].items():
                        if agent_name != 'final_analysis':
                            st.markdown(f"""
                                <div class="agent-card">
                                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                                        <span style="font-size: 1.5rem;">
                                            {get_agent_emoji(agent_name)}
                                        </span>
                                        <strong>{agent_name.replace('_', ' ').title()}</strong>
                                    </div>
                                    <div style="margin: 0.5rem 0;">
                                        {response.content}
                                    </div>
                                    <div class="metadata-section">
                                        <div>Response Confidence: {response.confidence:.2%}</div>
                                        <div>Analysis Time: {response.processing_time:.2f}s</div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {message['content']}
                    </div>
                """, unsafe_allow_html=True)
    
    # Enhanced chat input
    if prompt := st.chat_input("Describe your emergency situation for immediate assistance..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                async def process_query():
                    return await st.session_state.agent.process_query(
                        prompt,
                        st.session_state.agent_status.update_status
                    )
                
                responses = asyncio.run(process_query())
                
                if responses:
                    response_placeholder.markdown(f"""
                        <div class="chat-message assistant">
                            {responses['final_analysis'].content}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": responses
                    })
                
            except Exception as e:
                response_placeholder.error(f"Error processing emergency response: {str(e)}")

def get_agent_emoji(agent_name: str) -> str:
    """Get appropriate emoji for each agent type"""
    emoji_map = {
        'medical_response': '🏥',
        'personal_safety': '🛡️',
        'disaster_response': '🌪️',
        'women_safety': '👩',
        'final_analysis': '🚨'
    }
    return emoji_map.get(agent_name, '🤖')

if __name__ == "__main__":
    main()
