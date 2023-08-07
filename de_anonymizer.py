import os
import langchain
import pandas as pd
from conversations.conversation_handler import ConversationHandler

from utils import get_local_keys, load_google_search_tool, load_model


class DeAnonymizer:
    """
    Class of a de-anonimiser.
    """

    def __init__(self, llm_name: str, self_guide: bool = False, google: bool = False, debug: bool = False, verbose: bool = False, process_id=1):
        """
        Create a new instance of a de-anonymiser.
        :param llm: The LLM to use.
        """
        self.process_id = process_id
        # Accesses and keys
        langchain.debug = debug
        langchain.verbose = verbose
        self.llm_name = llm_name
        keys = get_local_keys()
        # os.environ["HUGGINGFACEHUB_API_TOKEN"] = keys["huggingface_hub_token"]
        os.environ["OPENAI_API_KEY"] = keys["openai_api_key"]

        # Define the LLM and the conversation handler
        self.llm = load_model(self.llm_name)
        self.conversation_handler = ConversationHandler(self.llm)

        # Define self-guide
        self.self_guide = self_guide
        # Define the google search tool
        self.google = load_google_search_tool() if google else None


    def re_identify(self, anon_text):
        self.conversation_handler.start_conversation(self.process_id)
        response = self.conversation_handler.send_new_message(user_input=anon_text)

        # if response.name != "Adele":
        #     # keep asking for the name
        #     response = self.conversation_handler.send_new_message()
        #     print("Hi")
        #     print(response.personas_1)
            # char1_names = ""
            # char2_names = ""
            # char3_names = ""
            # for i in range(5):
            #     char1_names += f"{response.personas_1[i]}, "
            #     char2_names += f"{response.personas_2[i]}, "
            #     char3_names += f"{response.personas_3[i]}, "

            # print(f"Personas 1:, {char1_names}") 
            # print(f"Personas 2:, {char2_names}") 
            # print(f"Personas 3:, {char3_names}") 

        self.conversation_handler.end_conversation()

        return response
    
    def re_identify_list(self, study_dir_path, file_names):
        """
            
        """
        res_columns = {
            1: ["Name", "Score", "Characteristic_1", "Characteristic_2", "Characteristic_3"], # process 1
        }

        df = pd.DataFrame(columns=res_columns[self.process_id])
        for i, file_name in enumerate(file_names):
            with open(os.path.join(study_dir_path, file_name), "r", encoding="utf-8") as f:
                anon_text = f.read()

            response = self.re_identify(anon_text)
            new_row = {
                "Name": response.name,
                "Score": response.score,
                "Characteristic_1": response.characteristics[0],
                "Characteristic_2": response.characteristics[1],
                "Characteristic_3": response.characteristics[2],
            }
            new_row_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_row_df], ignore_index=True)
            
        return df