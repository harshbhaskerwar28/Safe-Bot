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
    """Safety agent status management with sidebar display"""
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
            'working': '#28a745',
            'completed': '#17a2b8',
            'error': '#dc3545'
        }
        color = colors.get(status['status'], colors['idle'])
        
        st.markdown(f"""
            <div style="
                background-color: #1E1E1E;
                padding: 0.8rem;
                border-radius: 0.5rem;
                margin-bottom: 0.8rem;
                border: 1px solid {color};
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
    """Emergency response system with specialized safety agents"""
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
        """Initialize specialized safety agent prompts"""
        self.prompts = {
            'medical_response': """You are an Emergency Medical Response AI.
Query: {query}
Chat History: {chat_history}

Provide clear, step-by-step first aid guidance:
1. Immediate actions needed
2. Do's and Don'ts
3. When to seek professional help
4. Follow-up care steps

Focus on life-saving steps and clear instructions.""",

            'personal_safety': """You are a Personal Safety Response AI.
Query: {query}
Chat History: {chat_history}

Provide practical safety guidance:
1. Immediate safety steps
2. De-escalation techniques
3. Emergency contact advice
4. Prevention strategies

Keep instructions clear and actionable.""",

            'disaster_response': """You are a Disaster Response AI.
Query: {query}
Chat History: {chat_history}

Provide emergency response guidance:
1. Immediate safety measures
2. Evacuation instructions
3. Emergency supplies needed
4. Communication steps

Prioritize life-saving actions.""",

            'women_safety': """You are a Women's Safety Response AI.
Query: {query}
Chat History: {chat_history}

Provide focused safety guidance:
1. Immediate safety steps
2. Support resources
3. Legal rights and options
4. Prevention strategies

Maintain sensitivity while being practical."""
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
                    f"Safety analysis complete"
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
            raise Exception(f"Safety analysis error: {str(e)}")

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
            raise Exception(f"Safety agent {agent_name} error: {str(e)}")

    def _calculate_confidence(self, response: str) -> float:
        confidence = 0.7
        
        key_indicators = [
            "immediately", "urgent", "emergency", "call 911",
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
                    content="Hello! I'm here to help you with any emergency or safety situation. How can I assist you?",
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
            Analyze these safety recommendations and provide:

            1. IMMEDIATE ACTIONS (Most urgent steps)
            2. KEY SAFETY POINTS (Important precautions)
            3. WHEN TO GET HELP (Professional assistance criteria)
            4. FOLLOW-UP STEPS (After immediate crisis)

            Safety Assessments:
            {formatted_responses}
            
            Provide clear, actionable steps. Prioritize life-saving measures.
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
            raise Exception(f"Safety synthesis error: {str(e)}")

def setup_streamlit_ui():
    st.set_page_config(
        page_title="Emergency Safety Response System",
        page_icon="🚨",
        layout="wide"
    )
    
    st.markdown("""
        <style>
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
        }
        
        .agent-card {
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid #28a745;
            border-radius: 0.5rem;
            background-color: rgba(40, 167, 69, 0.05);
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
        }
        
        .emergency-button:hover {
            background-color: #c82333;
        }
        
        [data-testid="stSidebar"] {
            background-color: #1a1a1a;
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
            <h1>🚨 Emergency Safety Response System</h1>
            <p>Get immediate guidance for emergency situations</p>
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
                🚑 Emergency: Call 100,108
            </div>
            
            # Quick access buttons for common emergencies
            <div style='margin-top: 2rem;'>
                <h4>Quick Access Situations:</h4>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("🏥 Medical Emergency"):
            st.session_state.messages.append({
                "role": "user",
                "content": "What should I do in a medical emergency? Give me step by step instructions."
            })
        
        if st.button("🛡️ Personal Safety"):
            st.session_state.messages.append({
                "role": "user",
                "content": "I feel unsafe in my current situation. What immediate steps should I take?"
            })
            
        if st.button("🌪️ Natural Disaster"):
            st.session_state.messages.append({
                "role": "user",
                "content": "How should I prepare and respond to a natural disaster in my area?"
            })
            
        if st.button("👩 Women's Safety"):
            st.session_state.messages.append({
                "role": "user",
                "content": "What immediate steps should I take if I feel threatened or unsafe as a woman?"
            })
        
        st.markdown("""
            <div style='color: #28a745; margin-top: 2rem;'>
                <h3>🤖 Safety Agents Status</h3>
            </div>
        """, unsafe_allow_html=True)
        st.session_state.agent_status.initialize_sidebar_placeholder()
    
    # Main chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], dict):
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {message['content']['final_analysis'].content}
                    </div>
                """, unsafe_allow_html=True)
                
                with st.expander("🔍 Detailed Safety Analysis", expanded=False):
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
                                        <div>Confidence: {response.confidence:.2%}</div>
                                        <div>Response Time: {response.processing_time:.2f}s</div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message {message['role']}">
                        {message['content']}
                    </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Describe your emergency situation..."):
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
