setup_local:
	ansible-playbook for_spark.yml --connection=local --ask-sudo-pass
get_ansible_deps:
	ansible-galaxy install -fr requirements.yml --roles-path ./ansible-deps

.PHONY: get_ansible_deps setup_local
