💳 Pro Loan Approval Dashboard
An AI-powered financial assessment tool built with Streamlit. This dashboard goes beyond traditional credit scoring by combining Machine Learning predictive models with real-world financial business rules (DTI, CUR, and Affordability caps) to ensure responsible lending.

🚀 Key Features
Hybrid Approval Engine: Combines ML-generated credit scores with strict financial guardrails.

Real-time Risk Assessment: Dynamic calculation of Debt-to-Income (DTI) and Credit Utilization (CUR) ratios.

Predictive Scoring: Integrated with a custom ML scoring_service to evaluate borrower reliability.

Explainable AI (XAI): A detailed "Model Breakdown" section explaining the factors behind every approval or rejection.

Dynamic Visuals: Interactive Plotly gauge charts and metric cards that adapt to the loan's status.

Custom UI/UX: Responsive design with support for Dark and Light themes via CSS injection.

🧠 Decision Logic & Guardrails
To prevent predatory lending and ensure borrower stability, the dashboard employs a three-tier validation process:

ML Predictive Score: The system fetches a score (300–850). A score below 650 triggers an automatic rejection.

Financial Ratio Hard-Stops:

DTI (Debt-to-Income): Must be ≤40%. Even a high-score borrower is rejected if the debt burden is too high.

CUR (Credit Utilization): Must be ≤30%. Over-leveraged borrowers are flagged as high-risk.

Affordability Check: Ensures the monthly repayment does not exceed 25% of the borrower's monthly income.

🛠️ Tech Stack
Component	Technology
Frontend	Streamlit
Visualizations	Plotly Graph Objects
Data Processing	Pandas, NumPy
ML Integration	Custom Python Scoring Service
Styling	CSS3 (Variable-based theming)
📂 Project Structure
Bash
├── app.py                # Main Streamlit application
├── scoring_service.py    # ML logic & predictive scoring engine
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
⚙️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/your-username/loan-approval-dashboard.git
cd loan-approval-dashboard
Install dependencies:

Bash
pip install -r requirements.txt
Run the application:

Bash
streamlit run app.py
🏗️ Future Roadmap: Smart Workspace Integration
The next phase of this project involves integrating Computer Vision (CV) to track construction workspace progression.

Daily Progression Tracking: Using object detection (YOLO) and segmentation to monitor site work.

Smart Disbursement: Automatically releasing loan installments based on ML-verified project milestones (e.g., foundation completed, roofing finished).
