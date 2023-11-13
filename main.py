import pickle
import rdflib

from NERModel import NERModel
from QuestionIntentNN import QuestionIntentNN


def replace_entities_with_labels(doc):
    text = doc.text
    for ent in doc.ents:
        text = text.replace(str(ent), ent.label_)
    return text


def load_or_parse_graph(graph_path='./14_graph.nt', cache_path='cached_graph.pkl'):
    """
    Lädt einen gecachten Graphen oder parst ihn neu und speichert ihn im Cache.

    Args:
    - graph_path (str): Pfad zur Graphen-Datei.
    - cache_path (str): Pfad zur Cache-Datei.

    Returns:
    - rdflib.Graph: Der geladene oder geparste Graph.
    """
    try:
        # Versuche, den gecachten Graphen zu laden
        with open(cache_path, 'rb') as f:
            graph = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError):
        # Wenn das Laden fehlschlägt, parst den Graphen neu
        graph = rdflib.Graph()
        graph.parse(graph_path, format='turtle')

        # Speichere den neu geparsten Graphen im Cache
        with open(cache_path, 'wb') as f:
            pickle.dump(graph, f)

    return graph


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("load Graph:")
    graph = load_or_parse_graph()
    print("Graph loaded!")
    nlp= NERModel(graph)
    nlp.train_nlp_model()

    nlp= NERModel.return_trained_model()
    question_intent_modle = QuestionIntentNN()
    question_intent_modle.trainModel()
    question_intent_modle = question_intent_modle.loadModel()

    questions = ["Who is the director of Inception?",
                 "Recommend me a movie of the genre Horror?",
                 "Can you recommend me a movie of the genre Animation?",
                 "The Dark Knight Rises, The Dark Knight are movies i like i wanna see a similar Movie",
                 "What Genre is the movie Inception?",
                 "When was the movie Inception released?",
                 "Did Christopher Nolan direct Inception?",
                 "Who is the screenwriter of Inception?",
                 "Was Inception released in 2010?",
                 "Which movie released in 2010: Jurassic Park, Forrest Gump or Titanic?",
                 "Recommend me a Horror Movie which was released in 2010?",
                 "What is the director of Inception?",
                 "What is the genre of Inception?"]
    for question in questions:
        ner = nlp(question)
        doc = ner.doc
        print("The question was: " + question)
        print("Enties recognizes:" + str(len(doc.ents)))
        for entities in doc.ents:
            print("Entity:" + entities.text + " lable: " + entities.label_)
        print(question_intent_modle.predict([replace_entities_with_labels(ner)]))
        print()
        print()

    print("NER model trained!")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
