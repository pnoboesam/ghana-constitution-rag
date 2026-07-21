from src.rag import answer_question

while True:
    question = input(">>>")
    
    if question.lower() == "quit":
        answer = 'GoodBye'
        break
    else:
        answer = answer_question(question)['answer']

    print(answer)
