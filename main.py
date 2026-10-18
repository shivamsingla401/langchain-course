from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
    Swami Vivekananda (/ˈswɑːmi ˌvɪveɪˈkɑːnəndə/listen) (12 January 1863 – 4 July 1902), born Narendranath Datta,[b] was an Indian Hindu monk, philosopher, author, religious teacher, and the chief disciple of the Indian mystic Ramakrishna.[4][5] Vivekananda was a major figure in the introduction of Vedanta and Yoga to the Western world during the late 19th century,[6][7][8] and is credited with raising interfaith awareness and elevating Hinduism to the status of a major world religion.[9]

    Vivekananda showed an early inclination towards religion and spirituality. At the age of 18, he met Ramakrishna and became his devoted disciple, and later took up the vows of a sannyasin (renunciate). Following Ramakrishna’s death, Vivekananda travelled extensively across the Indian subcontinent as a wandering monk, gaining first-hand knowledge of the often harsh living conditions endured by the Indian masses under then British India, he sought a way to alleviate their suffering by establishing social services but lacked capital. In 1893, he travelled to the United States to participate in the Parliament of the World's Religions in Chicago, where he delivered a landmark speech beginning with the words "Sisters and brothers of America...". His powerful message introduced Hindu spiritual thought and advocated for both religious tolerance as well as universal acceptance.[10][11] The speech made a profound impression; an American newspaper described him as "an orator by divine right and undoubtedly the greatest figure at the Parliament".[12]

    Following his success in Chicago, Vivekananda lectured widely across the United States, the United Kingdom, and continental Europe, disseminating the essential principles of Hindu philosophy. He established the Vedanta Society of New York and the Vedanta Society of San Francisco (now the Vedanta Society of Northern California),[13] both of which became the foundations for later Vedanta Societies in the West. In India, he founded the Ramakrishna Math, a monastic order for spiritual training, and the Ramakrishna Mission, dedicated to social services, education, and humanitarian work.[7]

    Vivekananda is widely regarded as one of the greatest modern Indian thinkers. He was a prominent philosopher, social reformer, and the most successful proponent of Vedanta philosophy abroad. He played a crucial role in the Hindu revivalist movement and contributed significantly to the rise and development of Indian nationalism in colonial India.[14] Celebrated as a patriotic saint, his birth anniversary is observed in India as National Youth Day.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="qwen3.5:4b")
    #llm = ChatOpenAI(temperature=0, model="gpt-5")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
