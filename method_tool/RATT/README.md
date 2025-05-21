# This is a migrated version of RATT

## Method of Calling

    ratt = RATT(model, task, num_agents, num_steps, final_output_mode)
    draft_cot, final_answer = ratt(Question)

 In this way,you can create an instance of the RATT class with the specified parameters: model, task, num_agents, num_steps, and final_output_mode. 

 At runtime, the method outputs the index number of the current iteration and the current cumulative answer.

 And in the end, the instance returns two values: the thought process and the final answer.

 * model : This parameter indicates the model you are using.

 * task : This parameter indicates the dataset you are using.

 * num_agents : This parameter indicates the number of agents you are using(usually 1,unless you want to run multiple agents in parallel)

 * num_steps : This parameter controls th number of iterations in the process of generating answers, and each step may further refine of modify what has been generated.

 * final_output_mode : Method to generate the final output: "combine_each_step" to integrate outputs from each step, "only_last_step" to use the output from the final step as the final output.

 ## Matters Need Attention

 This method calls the google search api and may need to update the `google_api_key` and `google_cse_id` in the function: `get_search`.
 
 The `GOOGLE_API_KEY` is for Google Search API. You can get it from [Google Cloud Platform](https://cloud.google.com/docs/authentication/getting-started).

 And If you need to customize your Google Search API, for example, to restrict your searches to a specific website (such as Wikipedia), you will need a `GOOGLE_CSE_ID`. You can get it from [Google Programmable Search Engine](https://developers.google.com/custom-search/v1/overview).