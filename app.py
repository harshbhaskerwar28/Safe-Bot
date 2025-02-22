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
            temperature=0.7,  # Increased for more creative responses
            model_name="mixtral-8x7b-32768",  # Using a more capable model
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        self.chat_history = []
        self._initialize_prompts()
        self.agents = self._initialize_agents()

    def _initialize_prompts(self):
        """Initialize specialized safety agent prompts"""
        base_prompt = """You are an expert emergency response system. Your role is to provide detailed, practical guidance for emergency situations.

USER QUERY: {query}
CHAT HISTORY: {chat_history}

Analyze the situation carefully and provide comprehensive guidance following this structure:

1. SITUATION ASSESSMENT
- Analyze the specific emergency details
- Identify immediate risks and threats
- Determine priority actions

2. IMMEDIATE ACTIONS
- List specific step-by-step instructions
- Include exact timing for each step
- Specify who should take what action

3. SAFETY MEASURES
- Detail specific safety protocols
- List required safety equipment
- Provide environmental safety tips

4. EMERGENCY CONTACTS
- List relevant emergency numbers
- Specify when to contact each service
- Include backup contact options

5. FOLLOW-UP STEPS
- Provide post-emergency guidance
- List monitoring requirements
- Specify recovery actions

FORMAT YOUR RESPONSE:
• Use clear, specific language
• Include exact measurements and timing
• Provide specific examples
• Reference actual emergency protocols
• Include relevant medical/safety terms

Remember: Your guidance could save lives. Be thorough and precise."""

        self.prompts = {
            'medical_response': base_prompt + """

For medical emergencies:
- Use standard medical protocols
- Reference specific first aid procedures
- Include vital signs monitoring
- Specify medication considerations
- Detail infection control measures""",

            'personal_safety': base_prompt + """

For personal safety:
- Include de-escalation techniques
- Detail self-defense options
- Specify escape routes planning
- Include surveillance awareness
- Detail communication protocols""",

            'disaster_response': base_prompt + """

For disaster response:
- Include evacuation protocols
- Detail shelter requirements
- Specify resource management
- Include weather considerations
- Detail communication plans""",

            'women_safety': base_prompt + """

For women's safety:
- Include specific defense strategies
- Detail support network activation
- Specify legal protection steps
- Include documentation guidance
- Detail recovery resources"""
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
            # Determine relevant agents based on query content
            relevant_agents = self._select_relevant_agents(query)
            
            agent_tasks = []
            for agent_name in relevant_agents:
                status_callback(agent_name, 'working', 0.2, f"Analyzing situation")
                agent_tasks.append(self._get_agent_response(
                    agent_name, query, chat_history
                ))

            agent_responses = await asyncio.gather(*agent_tasks)
            
            for agent_name, response in zip(relevant_agents, agent_responses):
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
            raise Exception(f"Safety analysis error: {str(e)}")

    def _select_relevant_agents(self, query: str) -> List[str]:
        """Select relevant agents based on query content"""
        query = query.lower()
        selected_agents = []
        
        # Medical keywords
        if any(word in query for word in ['medical', 'health', 'hurt', 'pain', 'injury', 'sick', 'blood', 'breathing', 'heart']):
            selected_agents.append('medical_response')
            
        # Personal safety keywords
        if any(word in query for word in ['safety', 'threat', 'danger', 'scared', 'attack', 'follow', 'suspicious']):
            selected_agents.append('personal_safety')
            
        # Disaster keywords
        if any(word in query for word in ['disaster', 'flood', 'fire', 'earthquake', 'storm', 'hurricane', 'evacuation']):
            selected_agents.append('disaster_response')
            
        # Women's safety keywords
        if any(word in query for word in ['woman', 'women', 'girl', 'harassment', 'stalking', 'assault']):
            selected_agents.append('women_safety')
            
        # If no specific category is detected, use all agents
        if not selected_agents:
            selected_agents = list(self.agents.keys())
            
        return selected_agents

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
        """Calculate response confidence based on content analysis"""
        base_confidence = 0.7
        
        # Key phrases that indicate high-quality response
        quality_indicators = {
            "critical": 0.05,
            "immediate": 0.05,
            "emergency": 0.04,
            "warning": 0.04,
            "step": 0.03,
            "safety": 0.03,
            "contact": 0.03,
            "call": 0.03
        }
        
        # Structure indicators
        structure_indicators = {
            "1.": 0.05,
            "2.": 0.05,
            "3.": 0.05,
            "4.": 0.05
        }
        
        confidence = base_confidence
        
        # Check for quality phrases
        for phrase, weight in quality_indicators.items():
            if phrase.lower() in response.lower():
                confidence += weight
                
        # Check for proper structure
        for marker, weight in structure_indicators.items():
            if marker in response:
                confidence += weight
                
        # Penalize very short responses
        if len(response) < 100:
            confidence -= 0.2
        
        return min(0.95, max(0.3, confidence))

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
                    content="""Welcome to the Emergency Safety Response System. I'm here to provide immediate guidance for any emergency situation. How can I help you today?

Please describe your situation in detail, including:
- The type of emergency you're facing
- Your current location and environment
- Any immediate risks or threats
- Available resources or help nearby
- Any medical conditions or special circumstances

I can assist with:
- Medical emergencies and first aid guidance
- Personal safety in threatening situations
- Natural disaster response and preparation
- Women's safety and support
- General emergency guidance

The more details you provide, the more specific and helpful my guidance will be.""",
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
            
            synthesis_template = """You are the Emergency Response Coordinator synthesizing expert analyses into actionable guidance.

EMERGENCY QUERY: {query}

EXPERT ANALYSES:
{formatted_responses}

Create a comprehensive response that:
1. Addresses the specific emergency situation
2. Provides clear, actionable steps
3. Includes relevant safety measures
4. Lists specific emergency contacts
5. Details follow-up actions

FORMAT:
• Start with most critical actions
• Use clear, specific language
• Include exact timings and measurements
• Reference proper procedures and protocols
• Maintain a calm, authoritative tone

Remember: This is a real emergency situation requiring detailed, accurate guidance."""

            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", synthesis_template)
            ])
            
            synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
            
            synthesis_response = await synthesis_chain.ainvoke({
                "query": query,
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

        .thinking-animation {
            display: flex;
            gap: 0.5rem;
            align-items: center;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(40, 167, 69, 0.1);
            border: 1px solid #28a745;
            margin-bottom: 1rem;
        }

        .dot {
            width: 8px;
            height: 8px;
            background-color: #28a745;
            border-radius: 50%;
            animation: bounce 1.5s infinite;
        }
        
        .dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes bounce {
            0%, 60%, 100% {
                transform: translateY(0);
            }
            30% {
                transform: translateY(-10px);
            }
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
                response_placeholder = st.empty()
                thinking_placeholder = st.empty()
                
                # Show thinking animation
                thinking_placeholder.markdown("""
                    <div class="thinking-animation">
                        <div>Analyzing emergency situation...</div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                """, unsafe_allow_html=True)
                
                async def process_query():
                    return await st.session_state.agent.process_query(
                        prompt,
                        st.session_state.agent_status.update_status
                    )
                
                responses = asyncio.run(process_query())
                
                # Remove thinking animation
                thinking_placeholder.empty()
                
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
                thinking_placeholder.empty()  # Clear thinking animation on error
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
