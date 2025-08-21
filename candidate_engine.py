"""
BotHire - AI Candidate Recommendation Engine
Using FREE Sentence Transformers for Semantic Matching
Created by Deepti Agrawal

Features:
- Sentence Transformers for semantic understanding (FREE!)
- TF-IDF for keyword importance (fallback)
- Advanced candidate parsing and analysis
- Interactive Streamlit web interface
- Support for multiple PDF, DOC, DOCX, and TXT files
- Compare multiple candidates simultaneously
- AI-generated candidate summaries with hiring benefits

Installation:
# Core requirements (required)
pip install streamlit sentence-transformers pandas numpy scikit-learn matplotlib seaborn plotly

# File format support (optional)
pip install PyPDF2 python-docx

Usage:
streamlit run candidate_engine.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Try to import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.error("Sentence Transformers not installed. Run: pip install sentence-transformers")

# Try to import document processing libraries
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOC_SUPPORT = True
except ImportError:
    DOC_SUPPORT = False

# Page Configuration
st.set_page_config(
    page_title="BotHire",
    page_icon="BH",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .candidate-card {
        border: 2px solid #e1e5e9;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease;
    }
    .candidate-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    .excellent-match { 
        border-left: 6px solid #28a745; 
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
    }
    .good-match { 
        border-left: 6px solid #17a2b8; 
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); 
    }
    .moderate-match { 
        border-left: 6px solid #ffc107; 
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
    }
    .limited-match { 
        border-left: 6px solid #dc3545; 
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid #e1e5e9;
    }
    .skill-tag {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .score-bar {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        transition: width 0.3s ease;
    }
    .top-candidate-analysis {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 3px solid #28a745;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.15);
    }
    .top-candidate-analysis h2, .top-candidate-analysis h3, .top-candidate-analysis h4 {
        color: #28a745;
    }
    .hire-recommendation {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    .comparison-table {
        margin: 1rem 0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stDataFrame {
        background: white;
        border-radius: 10px;
    }
    .candidate-summary {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 2px solid #007bff;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0, 123, 255, 0.1);
        line-height: 1.6;
    }
    .candidate-summary h2 {
        color: #007bff;
        margin-bottom: 1rem;
    }
    .candidate-summary h3 {
        color: #495057;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.5rem;
    }
    .candidate-summary ol, .candidate-summary ul {
        margin: 1rem 0;
        padding-left: 1.5rem;
    }
    .candidate-summary li {
        margin-bottom: 0.8rem;
        line-height: 1.5;
    }
    .positives-list {
        background: #f8fff8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .benefits-list {
        background: #fff8f0;
        border-left: 4px solid #fd7e14;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

def read_pdf_file(uploaded_file):
    """Extract text from PDF file"""
    if not PDF_SUPPORT:
        st.error("PDF support not available. Install PyPDF2: pip install PyPDF2")
        return ""
    
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {str(e)}")
        return ""

def read_docx_file(uploaded_file):
    """Extract text from DOCX file"""
    if not DOC_SUPPORT:
        st.error("DOCX support not available. Install python-docx: pip install python-docx")
        return ""
    
    try:
        doc = Document(uploaded_file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX file: {str(e)}")
        return ""

def read_uploaded_file(uploaded_file):
    """Read uploaded file based on its type"""
    if uploaded_file.type == "text/plain":
        return str(uploaded_file.read(), "utf-8")
    elif uploaded_file.type == "application/pdf":
        if not PDF_SUPPORT:
            st.error("PDF support not available. Please install: pip install PyPDF2")
            return ""
        return read_pdf_file(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        if not DOC_SUPPORT:
            st.error("DOCX support not available. Please install: pip install python-docx")
            return ""
        return read_docx_file(uploaded_file)
    elif uploaded_file.type == "application/msword":
        st.warning("DOC files require conversion. Please save as DOCX or TXT format, or install: pip install python-docx")
        return ""
    else:
        st.error(f"Unsupported file type: {uploaded_file.type}")
        return ""

class AIJobMatcher:
    """
    Advanced AI-powered job matching system using Sentence Transformers
    """
    
    def __init__(self):
        self.model = None
        self.tfidf_vectorizer = None
        self.model_name = 'all-MiniLM-L6-v2'
        self.model = self.load_model()
    
    @st.cache_resource
    def load_model(_self):
        """Load the Sentence Transformer model (cached for performance)"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            st.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            return None
        
        try:
            with st.spinner(f"Loading AI model ({_self.model_name})..."):
                # Try loading the model with better error handling
                model = SentenceTransformer(_self.model_name)
                
                # Test the model with a simple encoding
                test_embedding = model.encode(["test sentence"])
                
                return model
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            
            # Try alternative models
            alternative_models = ['paraphrase-MiniLM-L3-v2', 'all-MiniLM-L12-v2']
            
            for alt_model in alternative_models:
                try:
                    st.info(f"Trying alternative model: {alt_model}")
                    model = SentenceTransformer(alt_model)
                    test_embedding = model.encode(["test sentence"])
                    return model
                except:
                    continue
            
            st.warning("All model loading attempts failed. Using TF-IDF fallback.")
            return None
    
    def preprocess_text(self, text):
        """Clean text for better processing"""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\-\+\#\.]', ' ', text)
        return text
    
    def extract_candidate_name(self, candidate_text, index):
        """Extract candidate name using smart heuristics"""
        lines = [line.strip() for line in candidate_text.split('\n') if line.strip()]
        
        # Check if there's a filename header (from file upload)
        for line in lines[:2]:
            if line.startswith('--- ') and line.endswith(' ---'):
                # Extract filename and use as candidate name
                filename = line.replace('--- ', '').replace(' ---', '')
                # Remove file extension and clean up
                name = filename.rsplit('.', 1)[0].replace('_', ' ').replace('-', ' ')
                return name.title()
        
        # Original name extraction logic
        for i, line in enumerate(lines[:4]):
            skip_patterns = [
                'resume', 'cv', 'curriculum', 'vitae', '@', '.com', 'phone', 
                'email', 'address', 'linkedin', 'github', 'summary', 'objective'
            ]
            
            if any(pattern in line.lower() for pattern in skip_patterns):
                continue
            
            words = line.split()
            if (2 <= len(words) <= 4 and 
                len(line) < 60 and 
                not any(char.isdigit() for char in line) and
                not line.startswith(('•', '-', '*', '→'))):
                return line.title()
        
        return f"Candidate {index + 1}"
    
    def extract_skills(self, text):
        """Extract technical skills from text"""
        skills_db = {
            'Programming Languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 
                'ruby', 'go', 'rust', 'swift', 'kotlin', 'scala', 'r', 'matlab'
            ],
            'Web Technologies': [
                'react', 'angular', 'vue', 'nodejs', 'express', 'django', 'flask', 
                'spring', 'laravel', 'rails', 'asp.net', 'html', 'css', 'sass', 'less'
            ],
            'Databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 
                'cassandra', 'oracle', 'sqlite', 'dynamodb', 'neo4j'
            ],
            'Cloud & DevOps': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab', 
                'github', 'terraform', 'ansible', 'vagrant', 'ci/cd', 'devops'
            ],
            'Data & AI': [
                'machine learning', 'deep learning', 'ai', 'data science', 'tensorflow', 
                'pytorch', 'pandas', 'numpy', 'scikit-learn', 'tableau', 'power bi', 'spark'
            ],
            'Mobile': [
                'android', 'ios', 'react native', 'flutter', 'xamarin', 'cordova', 'ionic'
            ],
            'Soft Skills': [
                'leadership', 'management', 'communication', 'problem solving', 
                'team work', 'agile', 'scrum', 'project management'
            ]
        }
        
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in skills_db.items():
            category_skills = []
            for skill in skills:
                if skill in text_lower:
                    category_skills.append(skill)
            if category_skills:
                found_skills[category] = category_skills
        
        return found_skills
    
    def extract_experience_years(self, text):
        """Extract years of experience from text"""
        patterns = [
            r'(\d+)\+?\s*years?\s*(of\s*)?(experience|exp)',
            r'(\d+)\+?\s*yrs?\s*(of\s*)?(experience|exp)',
            r'(\d+)\+?\s*year\s*(experience|exp)',
            r'experience[:\s]+(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in\s*(software|development|programming)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return int(match.group(1))
        return None
    
    def compute_semantic_similarity(self, job_description, candidates):
        """Compute semantic similarity using Sentence Transformers"""
        if not self.model:
            st.error("AI model not available. Using fallback method.")
            return self.compute_tfidf_similarity(job_description, candidates)
        
        job_clean = self.preprocess_text(job_description)
        candidates_clean = [self.preprocess_text(candidate) for candidate in candidates]
        
        all_texts = [job_clean] + candidates_clean
        embeddings = self.model.encode(all_texts, show_progress_bar=True)
        
        job_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(job_embedding, candidate_embeddings)[0]
        return similarities
    
    def compute_tfidf_similarity(self, job_description, candidates):
        """Fallback TF-IDF similarity computation"""
        all_docs = [job_description] + candidates
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_docs)
        job_vector = tfidf_matrix[0:1]
        candidate_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(job_vector, candidate_vectors)[0]
        return similarities
    
    def generate_fit_analysis(self, candidate_text, job_description, similarity_score, candidate_skills, years_exp):
        """Generate detailed AI-powered fit analysis"""
        
        job_lower = job_description.lower()
        job_skills = self.extract_skills(job_description)
        
        is_senior_role = any(term in job_lower for term in ['senior', 'lead', 'principal', 'architect'])
        required_years = None
        year_match = re.search(r'(\d+)\+?\s*years?', job_lower)
        if year_match:
            required_years = int(year_match.group(1))
        
        matching_skills = {}
        total_matches = 0
        
        for job_category, job_skill_list in job_skills.items():
            if job_category in candidate_skills:
                matches = list(set(job_skill_list) & set(candidate_skills[job_category]))
                if matches:
                    matching_skills[job_category] = matches
                    total_matches += len(matches)
        
        exp_analysis = ""
        if years_exp and required_years:
            if years_exp >= required_years:
                exp_analysis = f"Excellent experience match ({years_exp} years vs {required_years} required)"
            elif years_exp >= required_years * 0.8:
                exp_analysis = f"Good experience level ({years_exp} years, close to {required_years} required)"
            else:
                exp_analysis = f"Below required experience ({years_exp} years vs {required_years} required)"
        elif years_exp:
            if is_senior_role and years_exp >= 5:
                exp_analysis = f"Strong experience ({years_exp} years, suitable for senior role)"
            elif years_exp >= 3:
                exp_analysis = f"Good experience level ({years_exp} years)"
            else:
                exp_analysis = f"Limited experience ({years_exp} years)"
        else:
            exp_analysis = "Experience not specified in resume"
        
        skill_analysis = ""
        if total_matches >= 5:
            skill_analysis = f"Excellent skill alignment - {total_matches} matching skills across multiple areas"
        elif total_matches >= 3:
            skill_analysis = f"Good skill match - {total_matches} relevant skills identified"
        elif total_matches >= 1:
            skill_analysis = f"Some relevant skills - {total_matches} matches found"
        else:
            skill_analysis = "Limited skill overlap - few direct matches found"
        
        if similarity_score >= 0.7:
            level = "Excellent Match"
            css_class = "excellent-match"
            recommendation = "Strongly recommend for immediate interview - exceptional fit across all criteria"
        elif similarity_score >= 0.5:
            level = "Good Fit"
            css_class = "good-match"
            recommendation = "Recommend for interview - solid alignment with role requirements"
        elif similarity_score >= 0.3:
            level = "Moderate Match"
            css_class = "moderate-match"
            recommendation = "Consider for interview - potential fit with some gaps to address"
        else:
            level = "Limited Fit"
            css_class = "limited-match"
            recommendation = "Not recommended - significant gaps in required qualifications"
        
        summary = f"""
        **{level} ({similarity_score:.1%} semantic similarity)**
        
        {exp_analysis}
        
        {skill_analysis}
        
        **Key Matching Skills:**
        """
        
        for category, skills in matching_skills.items():
            if skills:
                skills_str = ", ".join(skills[:4])
                summary += f"\n• **{category}:** {skills_str}"
        
        summary += f"\n\n**Final Recommendation:** {recommendation}"
        
        return {
            'level': level,
            'summary': summary,
            'css_class': css_class,
            'matching_skills': matching_skills,
            'total_skill_matches': total_matches,
            'experience_analysis': exp_analysis
        }

def generate_candidate_summary(candidate_name, candidate_text, job_description, similarity_score, candidate_skills, years_exp, fit_analysis):
    """
    Generate comprehensive AI summary describing why this person is a great fit for the role
    Lists positives and benefits of hiring this candidate
    """
    
    job_lower = job_description.lower()
    is_senior_role = any(term in job_lower for term in ['senior', 'lead', 'principal', 'architect'])
    
    # Extract unique strengths and positives
    positives = []
    hiring_benefits = []
    
    # Experience-based positives
    if years_exp:
        if years_exp >= 8:
            positives.append(f"**Seasoned Professional**: {years_exp} years of proven industry experience")
            hiring_benefits.append("Brings deep expertise and can mentor junior team members")
        elif years_exp >= 5:
            positives.append(f"**Experienced Contributor**: {years_exp} years of solid professional experience")
            hiring_benefits.append("Can work independently and take ownership of complex projects")
        elif years_exp >= 2:
            positives.append(f"**Growing Professional**: {years_exp} years of relevant experience with strong fundamentals")
            hiring_benefits.append("Eager to grow and contribute with fresh perspectives")
        else:
            positives.append(f"**Fresh Talent**: {years_exp} year(s) of experience with enthusiasm to learn")
            hiring_benefits.append("Brings energy and modern approaches to traditional challenges")
    
    # Skill-based positives
    total_skills = sum(len(skill_list) for skill_list in candidate_skills.values())
    if total_skills >= 10:
        positives.append(f"**Technology Polyglot**: Proficient in {total_skills}+ technologies across multiple domains")
        hiring_benefits.append("Can adapt to various projects and technology stacks quickly")
    elif total_skills >= 6:
        positives.append(f"**Well-Rounded Technical Skills**: Strong foundation with {total_skills} key technologies")
        hiring_benefits.append("Versatile contributor who can work across different technical areas")
    elif total_skills >= 3:
        positives.append(f"**Focused Expertise**: Specialized knowledge in {total_skills} core technologies")
        hiring_benefits.append("Deep expertise in key areas relevant to your needs")
    
    # Specific skill category analysis
    for category, skills in candidate_skills.items():
        if skills:
            if category == 'Programming Languages' and len(skills) >= 3:
                positives.append(f"**Multi-Language Developer**: Proficient in {', '.join(skills[:3])} and more")
                hiring_benefits.append("Can work on diverse codebases and integrate different systems")
            
            elif category == 'Cloud & DevOps' and len(skills) >= 2:
                positives.append(f"**Cloud-Native Expertise**: Strong in modern infrastructure ({', '.join(skills[:3])})")
                hiring_benefits.append("Can optimize deployment processes and reduce operational costs")
            
            elif category == 'Data & AI' and len(skills) >= 2:
                positives.append(f"**Data-Driven Professional**: Advanced analytics capabilities ({', '.join(skills[:3])})")
                hiring_benefits.append("Can extract insights from data to drive business decisions")
            
            elif category == 'Web Technologies' and len(skills) >= 3:
                positives.append(f"**Full-Stack Capabilities**: Comprehensive web development skills ({', '.join(skills[:3])})")
                hiring_benefits.append("Can handle both frontend and backend development needs")
            
            elif category == 'Mobile' and len(skills) >= 1:
                positives.append(f"**Mobile Development Expertise**: Skilled in {', '.join(skills)}")
                hiring_benefits.append("Can expand your reach to mobile platforms and increase user engagement")
            
            elif category == 'Soft Skills' and len(skills) >= 2:
                positives.append(f"**Strong Leadership & Communication**: Proven {', '.join(skills[:2])} abilities")
                hiring_benefits.append("Can bridge technical and business teams effectively")
    
    # Similarity-based positives
    if similarity_score >= 0.7:
        positives.append(f"**Perfect Job Alignment**: {similarity_score:.1%} semantic match with role requirements")
        hiring_benefits.append("Minimal training required - can contribute immediately from day one")
    elif similarity_score >= 0.5:
        positives.append(f"**Strong Role Fit**: {similarity_score:.1%} alignment with job requirements")
        hiring_benefits.append("Quick ramp-up time with focused onboarding in specific areas")
    elif similarity_score >= 0.3:
        positives.append(f"**Good Foundation**: {similarity_score:.1%} baseline alignment with growth potential")
        hiring_benefits.append("Solid foundation to build upon with targeted development")
    
    # Resume content analysis for additional positives
    candidate_lower = candidate_text.lower()
    
    # Industry experience indicators
    if any(term in candidate_lower for term in ['startup', 'entrepreneur']):
        positives.append("**Startup Experience**: Thrives in fast-paced, dynamic environments")
        hiring_benefits.append("Brings agility and innovation mindset to accelerate growth")
    
    if any(term in candidate_lower for term in ['enterprise', 'fortune', 'large scale']):
        positives.append("**Enterprise Experience**: Proven ability to work with complex, large-scale systems")
        hiring_benefits.append("Can handle enterprise-level challenges and scalability requirements")
    
    # Leadership indicators
    leadership_terms = ['lead', 'manage', 'mentor', 'architect', 'design', 'strategy']
    leadership_count = sum(1 for term in leadership_terms if term in candidate_lower)
    if leadership_count >= 3:
        positives.append("**Leadership Potential**: Multiple indicators of leadership and strategic thinking")
        hiring_benefits.append("Can grow into leadership roles and help scale the team")
    
    # Education and certifications
    if any(term in candidate_lower for term in ['phd', 'doctorate']):
        positives.append("**Advanced Academic Background**: PhD-level expertise and research capabilities")
        hiring_benefits.append("Brings deep theoretical knowledge and research-driven problem solving")
    elif any(term in candidate_lower for term in ['master', 'mba', 'ms', 'm.s.']):
        positives.append("**Advanced Education**: Master's level expertise with strong analytical skills")
        hiring_benefits.append("Combines practical experience with advanced theoretical knowledge")
    elif any(term in candidate_lower for term in ['bachelor', 'bs', 'b.s.', 'be', 'b.e.']):
        positives.append("**Strong Educational Foundation**: Solid academic background in relevant field")
        hiring_benefits.append("Well-rounded knowledge base with systematic problem-solving approach")
    
    # Certification indicators
    if any(term in candidate_lower for term in ['certified', 'certification', 'aws certified', 'azure certified']):
        positives.append("**Professional Certifications**: Industry-recognized credentials and continuous learning")
        hiring_benefits.append("Stays current with industry standards and best practices")
    
    # Innovation and problem-solving indicators
    if any(term in candidate_lower for term in ['patent', 'innovation', 'research', 'publication']):
        positives.append("**Innovation Track Record**: Proven ability to create and implement new solutions")
        hiring_benefits.append("Can drive innovation and competitive advantage through creative solutions")
    
    # Open source and community involvement
    if any(term in candidate_lower for term in ['github', 'open source', 'contributor', 'community']):
        positives.append("**Community Involvement**: Active in open source and developer communities")
        hiring_benefits.append("Brings external perspective and stays connected with industry trends")
    
    # Generate the comprehensive summary
    summary = f"""
## AI-Generated Summary: Why {candidate_name} is a Great Fit

**Overall Assessment**: {candidate_name} represents {'an exceptional' if similarity_score >= 0.6 else 'a strong' if similarity_score >= 0.4 else 'a potential'} candidate for this role with {'outstanding' if similarity_score >= 0.6 else 'solid' if similarity_score >= 0.4 else 'reasonable'} alignment to your requirements.

### Key Positives & Strengths
"""
    
    for i, positive in enumerate(positives[:6], 1):  # Limit to top 6 positives for better readability
        summary += f"\n{i}. {positive}\n"
    
    summary += f"""
### Benefits of Hiring {candidate_name}
"""
    
    for i, benefit in enumerate(hiring_benefits[:6], 1):  # Limit to top 6 benefits for better readability
        summary += f"\n{i}. {benefit}\n"
    
    # Add specific value proposition
    if similarity_score >= 0.6:
        value_prop = f"{candidate_name} offers immediate value with minimal onboarding time. Their skill set aligns perfectly with your needs, making them a low-risk, high-reward hire."
    elif similarity_score >= 0.4:
        value_prop = f"{candidate_name} brings solid value with focused training in specific areas. Their foundation is strong, and they can quickly adapt to your specific requirements."
    else:
        value_prop = f"{candidate_name} offers growth potential with the right mentorship. While there are skill gaps to address, their foundation and attitude suggest they can develop into a valuable team member."
    
    summary += f"""
### Value Proposition

{value_prop}

### Quick Stats

- **Job Alignment**: {similarity_score:.1%}
- **Experience Level**: {years_exp if years_exp else 'Not specified'} years  
- **Technical Skills**: {total_skills} identified
- **Recommendation**: {fit_analysis['level']}

---

**Bottom Line**: {candidate_name} {'should be fast-tracked for interviews' if similarity_score >= 0.6 else 'deserves serious consideration' if similarity_score >= 0.4 else 'could be a good cultural fit with additional training'} based on this comprehensive analysis.
"""
    
    return summary

def parse_candidates(candidates_text):
    """Parse candidate text into individual resumes"""
    candidates = re.split(r'(?:\n\s*[-=]{3,}\s*\n|\n\s*\n\s*\n|\n\s*#{3,}\s*\n)', candidates_text)
    
    cleaned_candidates = []
    for candidate in candidates:
        candidate = candidate.strip()
        if len(candidate) > 150:
            cleaned_candidates.append(candidate)
    
    return cleaned_candidates

def generate_top_candidate_analysis(top_candidate, job_description):
    """Generate detailed analysis of why the top candidate should be hired"""
    
    candidate_name = top_candidate['name']
    similarity = top_candidate['similarity']
    years_exp = top_candidate['years_experience']
    skills = top_candidate['skills']
    candidate_text = top_candidate['full_text']
    
    job_lower = job_description.lower()
    
    # Extract job requirements for comparison
    job_requirements = []
    is_senior_role = any(term in job_lower for term in ['senior', 'lead', 'principal', 'architect'])
    required_years = None
    year_match = re.search(r'(\d+)\+?\s*years?', job_lower)
    if year_match:
        required_years = int(year_match.group(1))
    
    # Analyze the candidate's unique strengths
    unique_strengths = []
    
    # Experience analysis
    if years_exp:
        if is_senior_role and years_exp >= 5:
            unique_strengths.append(f"**Proven Senior-Level Experience**: With {years_exp} years of experience, {candidate_name} meets the senior-level requirements and brings the maturity needed for complex decision-making and leadership responsibilities.")
        elif years_exp >= 3:
            unique_strengths.append(f"**Solid Professional Experience**: {years_exp} years of hands-on experience demonstrates proven ability to deliver results and handle increasing responsibilities.")
        
        if required_years and years_exp >= required_years:
            unique_strengths.append(f"**Exceeds Experience Requirements**: {candidate_name} surpasses the required {required_years} years with {years_exp} years of experience, indicating reduced learning curve and immediate productivity.")
    
    # Skill analysis with specific examples
    skill_count = sum(len(skill_list) for skill_list in skills.values())
    if skill_count >= 8:
        unique_strengths.append(f"**Comprehensive Technical Skill Set**: With {skill_count} technical skills identified, {candidate_name} demonstrates versatility and ability to work across multiple technology stacks.")
    elif skill_count >= 5:
        unique_strengths.append(f"**Well-Rounded Technical Background**: {skill_count} relevant technical skills show strong foundation and adaptability.")
    
    # Specific skill matches
    job_skills = AIJobMatcher().extract_skills(job_description)
    critical_matches = []
    
    for job_category, job_skill_list in job_skills.items():
        if job_category in skills:
            matches = list(set(job_skill_list) & set(skills[job_category]))
            if matches:
                critical_matches.extend(matches)
    
    if critical_matches:
        unique_strengths.append(f"**Direct Skill Alignment**: Perfect match in {len(critical_matches)} critical skills: {', '.join(critical_matches[:5])}{'...' if len(critical_matches) > 5 else ''}")
    
    # Semantic similarity insights
    if similarity >= 0.7:
        unique_strengths.append(f"**Exceptional Job Fit**: {similarity:.1%} semantic similarity indicates outstanding alignment with job requirements, suggesting minimal training time and high success probability.")
    elif similarity >= 0.5:
        unique_strengths.append(f"**Strong Overall Alignment**: {similarity:.1%} semantic match demonstrates solid understanding of role requirements and relevant background.")
    
    # Leadership and soft skills analysis
    leadership_indicators = ['lead', 'manage', 'mentor', 'architect', 'design', 'strategy', 'project']
    leadership_count = sum(1 for indicator in leadership_indicators if indicator in candidate_text.lower())
    
    if leadership_count >= 3:
        unique_strengths.append(f"**Leadership Potential**: Multiple leadership indicators in resume suggest ability to guide teams and make strategic decisions.")
    
    # Industry-specific insights
    if 'startup' in candidate_text.lower() or 'entrepreneur' in candidate_text.lower():
        unique_strengths.append("**Startup Agility**: Experience in fast-paced startup environments indicates adaptability and ability to wear multiple hats.")
    
    if 'fortune' in candidate_text.lower() or 'enterprise' in candidate_text.lower():
        unique_strengths.append("**Enterprise Experience**: Background with large-scale organizations demonstrates understanding of complex business processes.")
    
    # Educational background
    education_keywords = ['phd', 'master', 'mba', 'computer science', 'engineering', 'certification']
    education_matches = [keyword for keyword in education_keywords if keyword in candidate_text.lower()]
    
    if education_matches:
        unique_strengths.append(f"**Strong Educational Foundation**: Advanced education background with focus on relevant fields ensures solid theoretical knowledge.")
    
    # Risk mitigation factors
    risk_factors = []
    if years_exp and required_years and years_exp < required_years * 0.8:
        risk_factors.append(f"Experience gap: {years_exp} years vs {required_years} required")
    
    if skill_count < 3:
        risk_factors.append("Limited technical skill diversity shown in resume")
    
    if similarity < 0.4:
        risk_factors.append("Lower semantic alignment may indicate role mismatch")
    
    # Generate the detailed analysis
    analysis = f"""
    ## Top Candidate Recommendation: {candidate_name}
    
    ### HIRING RECOMMENDATION: PROCEED WITH IMMEDIATE INTERVIEW
    
    **Match Score**: {similarity:.1%} | **Experience**: {years_exp if years_exp else 'Not specified'} years | **Skills Identified**: {skill_count}
    
    ---
    
    ### Why {candidate_name} Should Be Your Next Hire
    
    """
    
    for i, strength in enumerate(unique_strengths, 1):
        analysis += f"\n**{i}.** {strength}\n"
    
    analysis += f"""
    
    ---
    
    ### Key Skill Highlights
    
    """
    
    for category, skill_list in skills.items():
        if skill_list:
            analysis += f"**{category}**: {', '.join(skill_list[:6])}\n\n"
    
    analysis += f"""
    
    ---
    
    ### Business Impact Potential
    
    **Immediate Value**:
    - {similarity:.1%} job alignment suggests minimal onboarding time
    - Existing skill set allows for quick contribution to current projects
    - Experience level indicates ability to work independently
    
    **Long-term Benefits**:
    - Diverse technical background enables cross-functional collaboration
    - Strong foundation for potential leadership roles
    - Adaptability demonstrated through varied skill acquisition
    
    ---
    
    ### Considerations & Next Steps
    
    """
    
    if risk_factors:
        analysis += "**Areas to Address in Interview**:\n"
        for risk in risk_factors:
            analysis += f"- {risk}\n"
        analysis += "\n"
    else:
        analysis += "**No Major Red Flags Identified** - Candidate shows strong alignment across all key criteria.\n\n"
    
    analysis += f"""
    **Recommended Interview Focus**:
    1. Technical deep-dive in primary skill areas
    2. Behavioral questions about experience with similar roles
    3. Cultural fit assessment for team dynamics
    4. Problem-solving scenarios relevant to current challenges
    
    **Confidence Level**: {'High' if similarity > 0.6 and skill_count >= 5 else 'Medium-High' if similarity > 0.4 else 'Medium'}
    
    ---
    
    ### Bottom Line
    
    {candidate_name} represents an exceptional candidate who combines relevant experience, strong technical skills, and excellent job alignment. The {similarity:.1%} semantic match indicates this candidate understands the role requirements and possesses the background to excel in this position.
    
    **Recommendation**: **Schedule interview immediately** - This candidate should be prioritized in your hiring pipeline.
    """
    
    return analysis

def create_comparison_table(candidate_analyses):
    """Create a comparison table for candidates"""
    comparison_data = []
    
    for analysis in candidate_analyses:
        comparison_data.append({
            'Rank': f"#{analysis['rank']}",
            'Candidate': analysis['name'],
            'Match Score': f"{analysis['similarity']:.1%}",
            'Experience': f"{analysis['years_exp']} years" if analysis['years_exp'] else "Not specified",
            'Skills Count': analysis['skill_count'],
            'Key Strength': analysis['strengths'][0] if analysis['strengths'] else "General alignment",
            'Main Concern': analysis['weaknesses'][0] if analysis['weaknesses'] else "None identified"
        })
    
    return pd.DataFrame(comparison_data)

def create_skill_visualization(all_skills):
    """Create visualization of skill distribution"""
    if not all_skills:
        return None
    
    skill_counts = {}
    for candidate_skills in all_skills:
        for category, skills in candidate_skills.items():
            for skill in skills:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    if not skill_counts:
        return None
    
    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    if not top_skills:
        return None
    
    skills, counts = zip(*top_skills)
    
    fig = px.bar(
        x=list(counts),
        y=list(skills),
        orientation='h',
        title="Most Common Skills Across Candidates",
        labels={'x': 'Number of Candidates', 'y': 'Skills'},
        color=list(counts),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def main():
    # Header with title and creator credit
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("BotHire")
    with col2:
        st.markdown("""
        <div style="text-align: right; margin-top: 1rem; color: #666; font-size: 0.8rem;">
            Created by<br><strong>Deepti Agrawal</strong>
        </div>
        """, unsafe_allow_html=True)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        st.error("**Please install Sentence Transformers:**")
        st.code("pip install sentence-transformers", language="bash")
        st.stop()
    
    # Initialize matcher
    matcher = AIJobMatcher()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Matching parameters
        st.subheader("Matching Parameters")
        max_candidates = st.slider("Max candidates to display", 1, 20, 10)
        similarity_threshold = st.slider("Minimum similarity threshold", 0.0, 0.6, 0.2, 0.05)
        
        # Advanced options
        with st.expander("Advanced Options"):
            show_scores = st.checkbox("Show detailed scores", value=True)
            show_skills = st.checkbox("Show skill analysis", value=True)
            show_charts = st.checkbox("Show visualizations", value=True)
            show_top_analysis = st.checkbox("Show detailed top candidate analysis", value=True)
            show_candidate_summaries = st.checkbox("Show AI candidate summaries", value=True)
        
        # Creator credit at bottom of sidebar
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.75rem; margin-top: 2rem;">
            <strong>BotHire</strong><br>
            Created by Deepti Agrawal
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Job Description")
        job_description = st.text_area(
            "Enter the job requirements:",
            height=350,
            help="Provide a detailed job description including requirements, responsibilities, and qualifications"
        )
    
    with col2:
        st.header("Candidate Resumes")
        
        # File upload option for multiple files
        uploaded_files = st.file_uploader(
            "Upload resume files (multiple files supported)",
            type=['txt', 'pdf', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Upload multiple resume files in TXT, PDF, or DOCX format. Each file will be treated as a separate candidate."
        )
        
        candidates_text = ""
        if uploaded_files:
            all_file_contents = []
            for uploaded_file in uploaded_files:
                file_content = read_uploaded_file(uploaded_file)
                if file_content:
                    # Add filename as header for each resume
                    file_header = f"\n\n--- {uploaded_file.name} ---\n"
                    all_file_contents.append(file_header + file_content)
            
            if all_file_contents:
                candidates_text = "\n\n---\n\n".join(all_file_contents)
                st.success(f"Successfully loaded {len(uploaded_files)} resume files")
        
        # Text area for manual input
        candidates_input = st.text_area(
            "Or paste candidate resumes here:",
            value=candidates_text,
            height=300,
            help="Separate each resume with '---' or leave blank lines between resumes"
        )
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button("Find Best Matches", type="primary", use_container_width=True):
            process_matching = True
        else:
            process_matching = False
    
    with col2:
        if st.button("Clear All", use_container_width=True):
            st.rerun()
    
    with col3:
        st.write("")  # Empty space
    
    # Processing and Results
    if process_matching:
        if not job_description.strip():
            st.error("Please enter a job description!")
            st.stop()
        
        if not candidates_input.strip():
            st.error("Please provide candidate resumes!")
            st.stop()
        
        with st.spinner("AI is analyzing candidates using advanced semantic matching..."):
            # Parse candidates
            candidates = parse_candidates(candidates_input)
            
            if not candidates:
                st.error("No valid candidates found. Please separate resumes with '---' or blank lines.")
                st.stop()
            
            # Compute semantic similarities
            similarities = matcher.compute_semantic_similarity(job_description, candidates)
            
            # Process each candidate
            results = []
            all_skills = []
            
            for i, (candidate, similarity) in enumerate(zip(candidates, similarities)):
                if similarity >= similarity_threshold:
                    name = matcher.extract_candidate_name(candidate, i)
                    skills = matcher.extract_skills(candidate)
                    years_exp = matcher.extract_experience_years(candidate)
                    
                    fit_analysis = matcher.generate_fit_analysis(
                        candidate, job_description, similarity, skills, years_exp
                    )
                    
                    # Generate AI summary for this candidate
                    ai_summary = generate_candidate_summary(
                        name, candidate, job_description, similarity, skills, years_exp, fit_analysis
                    )
                    
                    results.append({
                        'rank': 0,
                        'name': name,
                        'similarity': similarity,
                        'skills': skills,
                        'years_experience': years_exp,
                        'fit_analysis': fit_analysis,
                        'ai_summary': ai_summary,
                        'preview': candidate[:400] + "..." if len(candidate) > 400 else candidate,
                        'full_text': candidate
                    })
                    
                    all_skills.append(skills)
            
            # Sort results by similarity
            results.sort(key=lambda x: x['similarity'], reverse=True)
            for i, result in enumerate(results[:max_candidates]):
                result['rank'] = i + 1
            
            results = results[:max_candidates]
        
        if not results:
            st.warning(f"No candidates found above {similarity_threshold:.1%} similarity threshold.")
            st.info("Try lowering the threshold in the sidebar or check your resume format.")
            st.stop()
        
        # Display results
        st.header(f"Top {len(results)} Candidates")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #1f77b4; margin: 0;">{len(candidates)}</h3>
                <p style="margin: 0; color: #666;">Candidates Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #28a745; margin: 0;">{len(results)}</h3>
                <p style="margin: 0; color: #666;">Above Threshold</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            best_score = results[0]['similarity'] if results else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #17a2b8; margin: 0;">{best_score:.1%}</h3>
                <p style="margin: 0; color: #666;">Best Match</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_score = np.mean([r['similarity'] for r in results]) if results else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: #fd7e14; margin: 0;">{avg_score:.1%}</h3>
                <p style="margin: 0; color: #666;">Average Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Top Candidate Detailed Analysis
        if show_top_analysis and results:
            st.markdown('<div class="hire-recommendation">', unsafe_allow_html=True)
            st.markdown("### TOP PICK RECOMMENDATION")
            st.markdown(f"**{results[0]['name']}** - {results[0]['similarity']:.1%} Match")
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.spinner("Generating detailed analysis for top candidate..."):
                top_analysis = generate_top_candidate_analysis(results[0], job_description)
            
            st.markdown('<div class="top-candidate-analysis">', unsafe_allow_html=True)
            st.markdown(top_analysis)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
        
        # Individual candidate results
        for result in results:
            st.markdown(f"""
            <div class="candidate-card {result['fit_analysis']['css_class']}">
            """, unsafe_allow_html=True)
            
            # Candidate header
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"### #{result['rank']} {result['name']}")
                if result['years_experience']:
                    st.markdown(f"**Experience:** {result['years_experience']} years")
            
            with col2:
                # Similarity score with progress bar
                st.markdown(f"**Similarity Score**")
                st.markdown(f"""
                <div class="score-bar">
                    <div class="progress-fill" style="width: {result['similarity']*100}%"></div>
                </div>
                <strong style="color: #28a745;">{result['similarity']:.1%}</strong>
                """, unsafe_allow_html=True)
            
            # Fit analysis
            st.markdown(result['fit_analysis']['summary'])
            
            # AI-Generated Candidate Summary (NEW FEATURE)
            if show_candidate_summaries:
                st.markdown('<div class="candidate-summary">', unsafe_allow_html=True)
                st.markdown(result['ai_summary'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Skills display
            if show_skills and result['skills']:
                st.markdown("**Technical Skills Found:**")
                skills_html = ""
                for category, skills in result['skills'].items():
                    for skill in skills[:4]:
                        skills_html += f'<span class="skill-tag">{skill}</span>'
                
                if skills_html:
                    st.markdown(skills_html, unsafe_allow_html=True)
            
            # Resume preview
            with st.expander(f"View Resume Preview - {result['name']}"):
                st.text(result['preview'])
                
                if st.button(f"Show Full Resume", key=f"full_{result['rank']}"):
                    st.text_area(
                        "Complete Resume:",
                        result['full_text'],
                        height=300,
                        key=f"full_text_{result['rank']}"
                    )
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("---")
        
        # Comparison Table
        if len(results) >= 2:
            st.subheader("Quick Comparison")
            comparison_data = []
            
            for result in results:
                comparison_data.append({
                    'Rank': f"#{result['rank']}",
                    'Candidate': result['name'],
                    'Match Score': f"{result['similarity']:.1%}",
                    'Experience': f"{result['years_experience']} years" if result['years_experience'] else "Not specified",
                    'Skills Count': sum(len(skill_list) for skill_list in result['skills'].values()),
                    'Recommendation': result['fit_analysis']['level']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.markdown('<div class="comparison-table">', unsafe_allow_html=True)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        if show_charts and len(results) > 1:
            st.header("Analysis Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Similarity scores chart
                names = [r['name'] for r in results]
                scores = [r['similarity'] for r in results]
                
                fig = px.bar(
                    x=scores,
                    y=names,
                    orientation='h',
                    title="Candidate Similarity Scores",
                    labels={'x': 'Similarity Score', 'y': 'Candidates'},
                    color=scores,
                    color_continuous_scale='viridis'
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Skills distribution chart
                skill_fig = create_skill_visualization(all_skills)
                if skill_fig:
                    st.plotly_chart(skill_fig, use_container_width=True)
                else:
                    st.info("Not enough skill data for visualization")
        
        # Export results
        st.header("Export Results")
        
        # Create export dataframe
        export_data = []
        for result in results:
            skills_flat = []
            for category, skills in result['skills'].items():
                skills_flat.extend(skills)
            
            export_data.append({
                'Rank': result['rank'],
                'Name': result['name'],
                'Similarity Score': f"{result['similarity']:.1%}",
                'Years Experience': result['years_experience'] or 'Not specified',
                'Skills': ', '.join(skills_flat[:10]),
                'Fit Level': result['fit_analysis']['level'],
                'Total Skill Matches': result['fit_analysis']['total_skill_matches']
            })
        
        export_df = pd.DataFrame(export_data)
        
        col1, col2 = st.columns(2)
        with col1:
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name="candidate_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.dataframe(export_df, use_container_width=True)

if __name__ == "__main__":
    main()