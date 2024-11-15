import streamlit as st
from groq import Groq
import random

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

def main():
    """
    Main function to generate experimental recipes based on ingredients.
    """
    
    # Set up Groq API key
    groq_api_key = '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'

    st.title("Experimental Recipe Generator")
    st.write("Enter the ingredients you have, and I'll suggest a creative recipe!")

    # Sidebar customization options
    st.sidebar.title('Customization')
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )

    user_ingredients = st.text_area("Enter ingredients (comma-separated):")
    
    if 'recipes' not in st.session_state:
        st.session_state.recipes = []

    # Memory for conversation chain
    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)

    # Initialize Groq Langchain chat object
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    # Generate a recipe based on the ingredients
    if user_ingredients:
        ingredients_list = user_ingredients.split(',')

        system_prompt = "Generate a creative and unique recipe using only the following ingredients. Suggest a name for the recipe, the cooking method. And no other ingredient to be used."

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
                HumanMessagePromptTemplate.from_template(
                    "Ingredients: {ingredients_input}"
                ),
            ]
        )

        # Create conversation chain
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            memory=memory,
        )

        # Get the recipe suggestion
        recipe_suggestion = conversation.predict(ingredients_input=', '.join(ingredients_list))
        
        # Display the generated recipe
        st.session_state.recipes.append({'ingredients': ingredients_list, 'recipe': recipe_suggestion})
        st.write("Suggested Recipe:", recipe_suggestion)
    
    # Show all previous recipes
    if st.session_state.recipes:
        st.subheader("Previous Recipes")
        for recipe in st.session_state.recipes:
            st.write(f"Ingredients: {', '.join(recipe['ingredients'])}")
            st.write(f"Recipe: {recipe['recipe']}")

if __name__ == "__main__":
    main()