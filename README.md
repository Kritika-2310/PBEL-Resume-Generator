**Introduction:**

This application is an AI-powered tool that generates professional resumes and cover letters based on user-provided information and template documents. The system uses Retrieval-Augmented Generation (RAG) technology to maintain formatting consistency with your provided templates while personalizing the content for each user.

**Features:**

 Template-Based Generation: Creates documents that match your professional templates
 AI-Powered Personalization: Tailors content to individual users' information
 Job-Specific Cover Letters: Generates targeted cover letters based on job descriptions
 Local LLM Processing: Uses Gemini model for privacy-focused processing
 Responsive UI: Clean interface built with Gradio and designed with Firebase

**Technology Stack:**

 Backend: Python
 AI Framework: LangChain
 LLM: Gemini
 Embeddings: BAAI/bge-small-en-v1.5
 Vector Database: FAISS
 UI Framework: Gradio
 UI/UX Design: Firebase

**Examples**
**Input:**
 Personal Information: "Kritika Ojha... [education, experience, skills]"
 Job Description: "AI Engineer position requiring Python, TensorFlow..."
**Output:**
 Professional resume in your template format
 <img width="1385" height="902" alt="RESUME GENERATOR" src="https://github.com/user-attachments/assets/e648df8d-81b1-4990-896e-8d9939e6cbc6" />
 Targeted cover letter addressing the AI Engineer position
 <img width="1172" height="898" alt="COVER LETTER GENERATOR" src="https://github.com/user-attachments/assets/f09d0440-62ef-4e1e-be1e-496b24c8e901" />
