class OpenAIGenerateWithFunctionCallingTemplate(OpenAIGenerateTemplate):
    def __init__(
        self,
        role: str,
        functions: List[Tool],
        after_transform: Optional[List[Hook]] = None,
        before_transform: Optional[List[Hook]] = None,
        **kwargs: Any,
    ):
        super().__init__(role=role, **kwargs)
        self.functions = functions
        self.after_transform = after_transform or []
        self.before_transform = before_transform or []

    def render(self, state: State) -> Message:
        # Call the function calling API
        function_calling_api_result = self._call_function_calling_api(state)

        # Call the function with the arguments provided by the API
        function_result = self._call_function(
            state, function_calling_api_result.arguments
        )

        # Call the LLM with the result of the function
        message = self._call_llm(state, function_result)

        return message

    def _call_function_calling_api(self, state: State) -> FunctionCallingAPIResult:
        # Prepare the input for the function calling API
        function_calling_api_input = self._prepare_function_calling_api_input(state)

        # Call the function calling API
        function_calling_api_result = self._call_llm(
            state, function_calling_api_input, is_function_calling_api=True
        )

        return function_calling_api_result

    def _prepare_function_calling_api_input(
        self, state: State
    ) -> FunctionCallingAPIInput:
        # Prepare the input for the function calling API
        input_template = FunctionCallingAPIInput(
            function_name=self.functions[0].name,
            description=self.functions[0].description,
            parameters={
                arg.get_name(): arg.get_value_type().__name__
                for arg in self.functions[0].argument_types
            },
        )

        return input_template

    def _call_function(
        self, state: State, arguments: Dict[str, Any]
    ) -> ToolResult:
        # Find the function to call
        function = self._find_function(state)

        # Check if the arguments are valid
        arguments = check_arguments(arguments, function.argument_types)

        # Call the function
        result = function.call(arguments, state)

        return result

    def _find_function(self, state: State) -> Tool:
        # Find the function to call based on the role
        for function in self.functions:
            if function.name == state.current_template_id:
                return function

        raise ValueError("Function not found")

    def _call_llm(
        self,
        state: State,
        input_template: Union[FunctionCallingAPIInput, ToolResult],
        is_function_calling_api: bool = False,
    ) -> Message:
        # Prepare the input for the LLM
        input_message = self._prepare_input_message(state, input_template)

        # Call the LLM
        output_message = self._call_llm_with_input_message(state, input_message)

        # Process the output message
        output_message = self._process_output_message(
            state, output_message, is_function_calling_api
        )

        return output_message

    def _prepare_input_message(
        self,
        state: State,
        input_template: Union[FunctionCallingAPIInput, ToolResult],
    ) -> Message:
        # Render the input template with the input data
        input_content = self._render_template(state, input_template)

        # Create the input message
        input_message = Message(content=input_content, role=self.role)

        return input_message

    def _call_llm_with_input_message(
        self, state: State, input_message: Message
    ) -> Message:
        # Add the input message to the session history
        session_history = state.session_history.add_message(input_message)

        # Create the session
        session = Session(messages=session_history.messages)

        # Call the LLM
        output_message = self.model.send(
            parameters=self.parameters, session=session
        )

        return output_message

    def _process_output_message(
        self,
        state: State,
        output_message: Message,
        is_function_calling_api: bool,
    ) -> Message:
        # Add the output message to the session history
        session_history = state.session_history.add_message(output_message)

        # Update the state with the new session history
        state = state.copy(session_history=session_history)

        # Process the output message with hooks
        if is_function_calling_api:
            hooks = self.after_transform
        else:
            hooks = self.before_transform + self.after_transform

        for hook in hooks:
            output_message = hook.process(state, output_message)

        return output_message
