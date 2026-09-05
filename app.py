from src.rag import answer_question

while True:
    question = input("\nHuman: ")
    
    if question.lower() == "quit":
        answer = 'GoodBye'
        print(f"\nAI: {answer}")
        break
    else:
        answer = answer_question(question)['answer']

    print(f"\nAI: {answer}")
