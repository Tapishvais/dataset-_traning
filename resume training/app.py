from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# NBCC Code of Conduct data in JSON format
nbcc_code_of_conduct = {
    "introduction": {
        "description": (
            "The Securities and Exchange Board of India (SEBI) formulated the SEBI (Prohibition of Insider Trading) "
            "Regulations, 2015 ('Regulations') under the SEBI Act, 1992 ('the Act'). These regulations came into effect "
            "on 15th May, 2015, and have been amended periodically, the latest amendment being the SEBI (Prohibition of "
            "Insider Trading) (Amendment) Regulations, 2021. The regulations apply to all companies listed on Indian stock "
            "exchanges. Consequently, NBCC is required to update its Code of Conduct to regulate, monitor, and report "
            "trading by Insiders ('Code of Conduct' or 'the Code')."
        )
    },
    "objective": {
        "description": (
            "The existing 'NBCC-Code of Conduct to Regulate, Monitor and Report Trading by Insiders' is replaced by this new Code. "
            "The Code aims to ensure monitoring, timely reporting, and adequate disclosure by insiders and/or designated persons "
            "of the Company. It seeks to ensure transparency and fairness in dealings with stakeholders and adherence to all applicable "
            "laws and regulations. The Company also maintains a Code of Practices and Procedures for fair disclosure of Unpublished Price Sensitive Information (UPSI). "
            "In case of any inconsistency between the Code and the Regulations, the provisions of the Regulations shall prevail."
        )
    },
    "definitions": {
        "company": {
            "description": "Company refers to NBCC (India) Limited (NBCC)."
        },
        "compliance_officer": {
            "description": (
                "The Company Secretary is appointed as the Compliance Officer. "
                "The Compliance Officer, who is financially literate, is responsible for: "
                "compliance of policies and procedures, maintenance of records, monitoring adherence to rules for preserving unpublished price-sensitive information, "
                "monitoring trades and implementing codes specified in the regulations, overall supervision by the CFO of the Company. "
                "In the absence of the Company Secretary, the CFO may authorize a senior officer to act as the Compliance Officer."
            )
        },
        "connected_person": {
            "description": (
                "Designated Persons "
                "Any person associated with the Company in the past six months, directly or indirectly, who has access to unpublished price-sensitive information. "
                "Persons in the following categories are deemed connected unless proven otherwise:"
            ),
            "categories": [
                "Immediate relatives of connected persons specified above",
                "Holding company, associate company, or subsidiary company",
                "Intermediaries specified in section 12 of the Act or their employees or directors",
                "Investment company, trustee company, asset management company, or their employees or directors",
                "Officers of the stock exchange or clearing house or corporation",
                "Members of the board of trustees of a mutual fund or the board of directors of the asset management company of a mutual fund or their employees",
                "Members of the board of directors or employees of a public financial institution as defined in section 2(72) of the Companies Act, 2013",
                "Officials or employees of a self-regulatory organization recognized or authorized by the Board",
                "Bankers of the Company",
                "Concern, firm, trust, Hindu undivided family, company, or association of persons where a director of the Company or his immediate relative or banker holds more than ten percent of the holding or interest"
            ]
        }
    }
}

# Initialize the QA pipeline with BERT
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

def answer_question(query, context):
    answers = []
    
    # Convert the context to a string, ensuring each section is included
    context_str = ""
    for key, value in context.items():
        if isinstance(value, dict) and 'description' in value:
            context_str += f"{key}: {value['description']}\n"
        else:
            print(f"Warning: Missing or incorrect structure in section {key}")

    # Print the context for debugging
    print("Context String:", context_str)
    
    try:
        # Process the context in chunks if it's too long
        context_chunks = [context_str[i:i+1000] for i in range(0, len(context_str), 1000)]  # Splitting context into chunks of 1000 characters

        for chunk in context_chunks:
            result = qa_pipeline(question=query, context=chunk)
            answers.append(result['answer'])

        # Combine the answers to provide a more comprehensive response
        combined_answer = " ".join(answers)
        return combined_answer
    
    except Exception as e:
        print("Error in answer_question function:", str(e))
        raise e

# Endpoint for NLP queries
@app.route('/query', methods=['POST'])
def handle_query():
    try:
        query = request.json.get('query')
        if not query:
            raise ValueError("Query parameter is missing or empty.")
        
        context = nbcc_code_of_conduct  # Use the provided JSON data directly
        answer = answer_question(query, context)
        return jsonify({"answer": answer})
    
    except Exception as e:
        print("Error in handle_query function:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
