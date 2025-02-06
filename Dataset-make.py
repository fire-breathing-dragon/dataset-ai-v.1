from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# Get user input for rows and columns
number_of_rows = int(input("Enter the quantity of rows: "))
number_of_columns = int(input("Enter the quantity of columns: "))

# Initialize the LLaMA 2 model
model = ChatOllama(model="deepseek-r1:8b", temperature=0.1, max_tokens=750)

# Define the prompt template
prompt_template = PromptTemplate(
    template="""Generate a dataset in JSON format with the following details:
- Category: {user_input}
- Rows: {number_of_rows}
- Columns: {number_of_columns}

### Instructions:
1. Create {number_of_rows} rows and {number_of_columns} columns.
2. Column names should be "Column 1", "Column 2", ..., "Column {number_of_columns}".
3. Data should match the category: {user_input}.
4. Output only the JSON dataset—no explanations.

### Example for "math problems" (2 rows, 3 columns):
[
  {{"Column 1": "2 + 3 = ?", "Column 2": "5 - 1 = ?", "Column 3": "4 * 6 = ?"}},
  {{"Column 1": "10 / 2 = ?", "Column 2": "7 + 8 = ?", "Column 3": "9 - 3 = ?"}}
]

### Your Task:
Generate the dataset for: {user_input}.
""",
    input_variables=["user_input", "number_of_rows", "number_of_columns"]  # Specify required variables
)

# Function to get user input and generate a response
def get_response(user_input):
    # Format the prompt with all required variables
    formatted_prompt = prompt_template.format(
        user_input=user_input,
        number_of_rows=number_of_rows,
        number_of_columns=number_of_columns
    )
    
    # Invoke the model with the formatted prompt
    response_message = model.invoke(formatted_prompt)
    
    return response_message.content

# Main loop for continuous interaction
if __name__ == "__main__":
    print("What dataset would you like to create today?")
    
    while True:
        user_input = input("Create a dataset (or type 'exit' to quit): ")
        
        # Exit condition
        if user_input.lower() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break

        # Get response from the model
        response = get_response(user_input)
        print("AI Response:", response)