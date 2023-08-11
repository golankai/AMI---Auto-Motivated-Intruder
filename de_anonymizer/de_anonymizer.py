import os
import langchain
import pandas as pd
from ami_process.ami_process import AMI_process_handler
from conversations.conversation_handler import ConversationHandler

from utils import get_local_keys, load_google_search_tool, load_model


class DeAnonymizer:
    """
    Class of a de-anonimiser.
    """

    def __init__(
        self,
        llm_name: str,
        self_guide: bool = False,
        google: bool = False,
        debug: bool = False,
        verbose: bool = False,
        process_id=1,
    ):
        """
        Create a new instance of a de-anonymiser.
        :param llm: The LLM to use.
        """
        self.process_handler = AMI_process_handler(process_id)

        # Accesses and keys
        langchain.debug = debug
        langchain.verbose = verbose
        llm_name = llm_name
        keys = get_local_keys()
        # os.environ["HUGGINGFACEHUB_API_TOKEN"] = keys["huggingface_hub_token"]
        os.environ["OPENAI_API_KEY"] = keys["openai_api_key"]

        # Define the LLM and the conversation handler
        llm = load_model(llm_name)
        self.conversation_handler = ConversationHandler(llm)

        # Define self-guide
        self.self_guide = self_guide
        # Define the google search tool
        self.google = load_google_search_tool() if google else None


    def re_identify(self, anon_text, df=None, file_name=None):
        """
        Re-identify a single text.
        :param anon_text: The anonymized text.
        :df : The dataframe to save the results to. If None, the results are not saved.
        """
        self.conversation_handler.start_conversation(self.process_handler.get_base_template())
        self.process_handler.new_process()
        response = ""
        
        for query in self.process_handler:
            conv_responses_object = {}
            response = self.conversation_handler.send_new_message(query, user_input=anon_text)
            # update the process handler with the last response. So, it enables the process to decide whether to keep going or not. (based on the last response)
            self.process_handler.set_last_response(response) 

            conv_responses_object = response
            # currently, we support add_row only for one question.
            # TODO: support more than one question (add_row for all the questions of the process dataß)
            # for key, value in response.items():
            #     conv_responses_object[key] = value
        
        if df is not None:
            df = self.add_row_to_csv(df, conv_responses_object, file_name)
        else:
            print(response)    

        self.conversation_handler.end_conversation()
        return df
    

    def add_row_to_csv(self, df, conv_responses_object, file_name):
        new_row = self.process_handler.get_df_row(conv_responses_object, file_name)
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)
        return df
            
        
    def re_identify_list(self, study_dir_path, file_names, save_to_csv=False):
        df = None
        if save_to_csv:
            res_columns = self.process_handler.get_res_columns()
            df = pd.DataFrame(columns=res_columns)
        
        for i, file_name in enumerate(file_names):
            with open(
                os.path.join(study_dir_path, file_name), "r", encoding="utf-8"
            ) as f:
                anon_text = f.read()
            df = self.re_identify(anon_text, df, file_name)
            
        return df
