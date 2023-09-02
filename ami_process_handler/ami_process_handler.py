from process_data.process_utils import get_process_data


class AMI_process_handler:
    def __init__(self, process_id) -> None:
        self.process_id = process_id

        self.process_data = get_process_data(self.process_id)
        self.num_queries = len(self.process_data.queries)
        self.query_number = 0
        self.conv_responses = {}

    def new_process(self):
        self.query_number = 0
        self.conv_responses = {}

    def get_base_template(self):
        return self.process_data.get_base_template()

    def get_conv_responses(self):
        return self.conv_responses

    def __iter__(self):
        return self

    def set_last_response(self, last_response):
        self.conv_responses.update(last_response.dict())

    def __next__(self):
        if self.query_number >= self.num_queries:
            raise StopIteration

        query = self.process_data.queries[self.query_number]

        # I want that the ami process will decide if keep going or not by using the last response and the process_data  (handle_conditions)
        # self.process_data?.handle_conditions(self.query_number, self.last_response)

        if query is None:
            raise StopIteration

        self.query_number += 1
        return query
