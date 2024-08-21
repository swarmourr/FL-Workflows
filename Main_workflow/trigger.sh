echo Creating FL_ensemble if not exist 
pegasus-em create FL_ensemble2 2>/dev/null
echo Creating triggers if not exist 
pegasus-em file-pattern-trigger "FL_ensemble2" "local_training_$f2_$(echo $RANDOM | md5sum | head -c 20)" 10s "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/plan.sh" "/home/hsafri/FederatedLearning/federated-learning-score-EM-blog/Main_workflow/output/local_wf_round_*.yml" --timeout 90m


