/home/pululun/yandex/rasa_env/lib64/python3.9/site-packages/rasa/core/tracker_store.py:1044: MovedIn20Warning: Deprecated API features detected! These feature(s) are not compatible with SQLAlchemy 2.0. To prevent incompatible upgrades prior to updating applications, ensure requirements files are pinned to "sqlalchemy<2.0". Set environment variable SQLALCHEMY_WARN_20=1 to show all deprecation warnings.  Set environment variable SQLALCHEMY_SILENCE_UBER_WARNING=1 to silence this message. (Background on SQLAlchemy 2.0 at: https://sqlalche.me/e/b8d9)
Base: DeclarativeMeta = declarative_base()
usage: rasa run [-h] [-v] [-vv] [--quiet] [--logging-config-file LOGGING_CONFIG_FILE] [-m MODEL] [--log-file LOG_FILE] [--use-syslog] [--syslog-address SYSLOG_ADDRESS] [--syslog-port SYSLOG_PORT] [--syslog-protocol SYSLOG_PROTOCOL]
[--endpoints ENDPOINTS] [-i INTERFACE] [-p PORT] [-t AUTH_TOKEN] [--cors [CORS ...]] [--enable-api] [--response-timeout RESPONSE_TIMEOUT] [--request-timeout REQUEST_TIMEOUT] [--remote-storage REMOTE_STORAGE]
[--ssl-certificate SSL_CERTIFICATE] [--ssl-keyfile SSL_KEYFILE] [--ssl-ca-file SSL_CA_FILE] [--ssl-password SSL_PASSWORD] [--credentials CREDENTIALS] [--connector CONNECTOR] [--jwt-secret JWT_SECRET]
[--jwt-method JWT_METHOD] [--jwt-private-key JWT_PRIVATE_KEY]
{actions} ... [model-as-positional-argument]
rasa run: error: argument {actions}: invalid choice: 'DEBUG' (choose from 'actions')
(rasa_env) pululun@fedora:~/yandex/rasa$ rasa run --enable-api -vv
/home/pululun/yandex/rasa_env/lib64/python3.9/site-packages/rasa/core/tracker_store.py:1044: MovedIn20Warning: Deprecated API features detected! These feature(s) are not compatible with SQLAlchemy 2.0. To prevent incompatible upgrades prior to updating applications, ensure requirements files are pinned to "sqlalchemy<2.0". Set environment variable SQLALCHEMY_WARN_20=1 to show all deprecation warnings.  Set environment variable SQLALCHEMY_SILENCE_UBER_WARNING=1 to silence this message. (Background on SQLAlchemy 2.0 at: https://sqlalche.me/e/b8d9)
Base: DeclarativeMeta = declarative_base()
/home/pululun/yandex/rasa_env/lib64/python3.9/site-packages/rasa/shared/utils/validation.py:134: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
import pkg_resources
/home/pululun/yandex/rasa_env/lib64/python3.9/site-packages/pkg_resources/__init__.py:2868: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('mpl_toolkits')`.
Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
declare_namespace(pkg)
/home/pululun/yandex/rasa_env/lib64/python3.9/site-packages/pkg_resources/__init__.py:2868: DeprecationWarning: Deprecated call to `pkg_resources.declare_namespace('ruamel')`.
Implementing implicit namespace packages (as specified in PEP 420) is preferred to `pkg_resources.declare_namespace`. See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages
declare_namespace(pkg)
2024-10-23 22:59:46 DEBUG    rasa.cli.utils  - Parameter 'credentials' not set. Using default location 'credentials.yml' instead.
/home/pululun/yandex/rasa_env/lib64/python3.9/site-packages/sanic_cors/extension.py:39: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
SANIC_VERSION = LooseVersion(sanic_version)
2024-10-23 22:59:46 DEBUG    h5py._conv  - Creating converter from 7 to 5
2024-10-23 22:59:46 DEBUG    h5py._conv  - Creating converter from 5 to 7
2024-10-23 22:59:46 DEBUG    h5py._conv  - Creating converter from 7 to 5
2024-10-23 22:59:46 DEBUG    h5py._conv  - Creating converter from 5 to 7
2024-10-23 22:59:47 DEBUG    jax._src.path  - etils.epath was not found. Using pathlib for file I/O.
/home/pululun/yandex/rasa_env/lib64/python3.9/site-packages/tensorflow/lite/python/util.py:52: DeprecationWarning: jax.xla_computation is deprecated. Please use the AOT APIs.
from jax import xla_computation as _xla_computation
2024-10-23 22:59:47 DEBUG    rasa.core.utils  - Available web server routes:
/conversations/<conversation_id:path>/messages     POST                           add_message
/conversations/<conversation_id:path>/tracker/events POST                           append_events
/webhooks/rasa                                     GET                            custom_webhook_RasaChatInput.health
/webhooks/rasa/webhook                             POST                           custom_webhook_RasaChatInput.receive
/webhooks/rest                                     GET                            custom_webhook_RestInput.health
/webhooks/rest/webhook                             POST                           custom_webhook_RestInput.receive
/model/test/intents                                POST                           evaluate_intents
/model/test/stories                                POST                           evaluate_stories
/conversations/<conversation_id:path>/execute      POST                           execute_action
/domain                                            GET                            get_domain
/                                                  GET                            hello
/model                                             PUT                            load_model
/model/parse                                       POST                           parse
/conversations/<conversation_id:path>/predict      POST                           predict
/conversations/<conversation_id:path>/tracker/events PUT                            replace_events
/conversations/<conversation_id:path>/story        GET                            retrieve_story
/conversations/<conversation_id:path>/tracker      GET                            retrieve_tracker
/status                                            GET                            status
/model/predict                                     POST                           tracker_predict
/model/train                                       POST                           train
/conversations/<conversation_id:path>/trigger_intent POST                           trigger_intent
/model                                             DELETE                         unload_model
/version                                           GET                            version
2024-10-23 22:59:47 INFO     root  - Starting Rasa server on http://0.0.0.0:5005
2024-10-23 22:59:47 DEBUG    rasa.core.utils  - Using the default number of Sanic workers (1).
2024-10-23 22:59:47 DEBUG    rasa.telemetry  - Skipping telemetry reporting: no license hash found.
2024-10-23 22:59:47 DEBUG    rasa.core.tracker_store  - Connected to InMemoryTrackerStore.
2024-10-23 22:59:47 DEBUG    rasa.core.lock_store  - Connected to lock store 'InMemoryLockStore'.
2024-10-23 22:59:47 DEBUG    rasa.core.nlg.generator  - Instantiated NLG to 'TemplatedNaturalLanguageGenerator'.
2024-10-23 22:59:47 INFO     rasa.core.processor  - Loading model models/20241023-224641-excited-service.tar.gz...
2024-10-23 22:59:47 DEBUG    rasa.engine.storage.local_model_storage  - Extracted model to '/tmp/tmpoyrkfrgj'.
2024-10-23 22:59:47 DEBUG    rasa.engine.graph  - Node 'nlu_message_converter' loading 'NLUMessageConverter.load' and kwargs: '{}'.
2024-10-23 22:59:47 DEBUG    rasa.engine.graph  - Node 'run_WhitespaceTokenizer0' loading 'WhitespaceTokenizer.load' and kwargs: '{}'.
2024-10-23 22:59:47 DEBUG    rasa.engine.graph  - Node 'run_RegexFeaturizer1' loading 'RegexFeaturizer.load' and kwargs: '{}'.
2024-10-23 22:59:47 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'train_RegexFeaturizer1' was requested for reading.
2024-10-23 22:59:47 DEBUG    rasa.engine.graph  - Node 'run_LexicalSyntacticFeaturizer2' loading 'LexicalSyntacticFeaturizer.load' and kwargs: '{}'.
2024-10-23 22:59:47 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'train_LexicalSyntacticFeaturizer2' was requested for reading.
2024-10-23 22:59:47 DEBUG    rasa.engine.graph  - Node 'run_CountVectorsFeaturizer3' loading 'CountVectorsFeaturizer.load' and kwargs: '{}'.
2024-10-23 22:59:47 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'train_CountVectorsFeaturizer3' was requested for reading.
2024-10-23 22:59:47 DEBUG    rasa.engine.graph  - Node 'run_LanguageModelFeaturizer4' loading 'LanguageModelFeaturizer.load' and kwargs: '{}'.
2024-10-23 22:59:47 DEBUG    rasa.nlu.featurizers.dense_featurizer.lm_featurizer  - Loading Tokenizer and Model for bert
2024-10-23 22:59:47 DEBUG    urllib3.connectionpool  - Starting new HTTPS connection (1): huggingface.co:443
2024-10-23 22:59:53 DEBUG    urllib3.connectionpool  - https://huggingface.co:443 "HEAD /bert-base-cased/resolve/main/tokenizer_config.json HTTP/1.1" 200 0
2024-10-23 22:59:53 DEBUG    urllib3.connectionpool  - https://huggingface.co:443 "HEAD /bert-base-cased/resolve/main/config.json HTTP/1.1" 200 0
Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFBertModel: ['cls.seq_relationship.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.bias', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight']
- This IS expected if you are initializing TFBertModel from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TFBertModel from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
  All the weights of TFBertModel were initialized from the PyTorch model.
  If your task is similar to the task the model of the checkpoint was trained on, you can already use TFBertModel for predictions without further training.
  2024-10-23 22:59:54 DEBUG    rasa.engine.graph  - Node 'run_DIETClassifier5' loading 'DIETClassifier.load' and kwargs: '{}'.
  2024-10-23 22:59:54 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'train_DIETClassifier5' was requested for reading.
  2024-10-23 22:59:54 DEBUG    rasa.utils.tensorflow.models  - Loading the model from /tmp/tmpx70x0on1/train_DIETClassifier5/DIETClassifier.tf_model with finetune_mode=False...
  2024-10-23 22:59:54 DEBUG    rasa.nlu.classifiers.diet_classifier  - You specified 'DIET' to train entities, but no entities are present in the training data. Skipping training of entities.
  2024-10-23 22:59:54 DEBUG    rasa.nlu.classifiers.diet_classifier  - Following metrics will be logged during training:
  2024-10-23 22:59:54 DEBUG    rasa.nlu.classifiers.diet_classifier  -   t_loss (total loss)
  2024-10-23 22:59:54 DEBUG    rasa.nlu.classifiers.diet_classifier  -   i_acc (intent acc)
  2024-10-23 22:59:54 DEBUG    rasa.nlu.classifiers.diet_classifier  -   i_loss (intent loss)
  2024-10-23 23:00:05 DEBUG    rasa.utils.tensorflow.models  - Finished loading the model.
  /home/pululun/yandex/rasa_env/lib64/python3.9/site-packages/rasa/utils/train_utils.py:530: UserWarning: constrain_similarities is set to `False`. It is recommended to set it to `True` when using cross-entropy loss.
  rasa.shared.utils.io.raise_warning(
  2024-10-23 23:00:05 DEBUG    rasa.engine.graph  - Node 'run_EntitySynonymMapper6' loading 'EntitySynonymMapper.load' and kwargs: '{}'.
  2024-10-23 23:00:05 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'train_EntitySynonymMapper6' was requested for reading.
  2024-10-23 23:00:05 DEBUG    rasa.nlu.extractors.entity_synonyms  - Failed to load ABCMeta from model storage. Resource 'train_EntitySynonymMapper6' doesn't exist.
  2024-10-23 23:00:05 DEBUG    rasa.engine.graph  - Node 'run_ResponseSelector7' loading 'ResponseSelector.load' and kwargs: '{}'.
  2024-10-23 23:00:05 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'train_ResponseSelector7' was requested for reading.
  2024-10-23 23:00:05 DEBUG    rasa.nlu.classifiers.diet_classifier  - Failed to load ABCMeta from model storage. Resource 'train_ResponseSelector7' doesn't exist.
  2024-10-23 23:00:05 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'train_ResponseSelector7' was requested for reading.
  2024-10-23 23:00:05 DEBUG    rasa.nlu.selectors.response_selector  - Failed to load ResponseSelector from model storage. Resource 'train_ResponseSelector7' doesn't exist.
  2024-10-23 23:00:05 DEBUG    rasa.engine.graph  - Node 'run_RegexMessageHandler' loading 'RegexMessageHandler.load' and kwargs: '{}'.
  2024-10-23 23:00:05 DEBUG    rasa.engine.graph  - Node 'domain_provider' loading 'DomainProvider.load' and kwargs: '{}'.
  2024-10-23 23:00:05 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'domain_provider' was requested for reading.
  2024-10-23 23:00:05 DEBUG    rasa.engine.graph  - Node 'run_MemoizationPolicy0' loading 'MemoizationPolicy.load' and kwargs: '{}'.
  2024-10-23 23:00:05 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'train_MemoizationPolicy0' was requested for reading.
  2024-10-23 23:00:05 DEBUG    rasa.engine.graph  - Node 'run_RulePolicy1' loading 'RulePolicy.load' and kwargs: '{}'.
  2024-10-23 23:00:05 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'train_RulePolicy1' was requested for reading.
  2024-10-23 23:00:05 DEBUG    rasa.engine.graph  - Node 'run_TEDPolicy2' loading 'TEDPolicy.load' and kwargs: '{}'.
  2024-10-23 23:00:05 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'train_TEDPolicy2' was requested for reading.
  2024-10-23 23:00:05 DEBUG    rasa.utils.tensorflow.models  - Loading the model from /tmp/tmpx70x0on1/train_TEDPolicy2/ted_policy.tf_model with finetune_mode=False...
  2024-10-23 23:00:10 DEBUG    rasa.utils.tensorflow.models  - Finished loading the model.
  2024-10-23 23:00:10 DEBUG    rasa.engine.graph  - Node 'rule_only_data_provider' loading 'RuleOnlyDataProvider.load' and kwargs: '{}'.
  2024-10-23 23:00:10 DEBUG    rasa.engine.storage.local_model_storage  - Resource 'train_RulePolicy1' was requested for reading.
  2024-10-23 23:00:10 DEBUG    rasa.engine.graph  - Node 'select_prediction' loading 'DefaultPolicyPredictionEnsemble.load' and kwargs: '{}'.
  2024-10-23 23:00:10 INFO     root  - Rasa server is up and running.
  2024-10-23 23:00:10 INFO     root  - Enabling coroutine debugging. Loop id 94515852424016.
  2024-10-23 23:01:13 DEBUG    rasa.core.lock_store  - Issuing ticket for conversation 'name'.
  2024-10-23 23:01:13 DEBUG    rasa.core.lock_store  - Acquiring lock for conversation 'name'.
  2024-10-23 23:01:13 DEBUG    rasa.core.lock_store  - Acquired lock for conversation 'name'.
  2024-10-23 23:01:13 DEBUG    rasa.core.tracker_store  - Could not find tracker for conversation ID 'name'.
  2024-10-23 23:01:13 DEBUG    rasa.core.tracker_store  - No event broker configured. Skipping streaming events.
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - Starting a new session for conversation ID 'name'.
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[]
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=action_session_start rasa_events=[<rasa.shared.core.events.SessionStarted object at 0x7fb2e1a876d0>, ActionExecuted(action: action_listen, policy: None, confidence: None)]
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - [debug    ] processor.slots.log            slot_values=	session_started_metadata: None
  2024-10-23 23:01:13 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__message__': [<rasa.core.channels.channel.UserMessage object at 0x7fb2e00098b0>], '__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2dbff6850>}, targets: ['run_RegexMessageHandler'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'nlu_message_converter' running 'NLUMessageConverter.convert_user_message'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_WhitespaceTokenizer0' running 'WhitespaceTokenizer.process'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_RegexFeaturizer1' running 'RegexFeaturizer.process'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_LexicalSyntacticFeaturizer2' running 'LexicalSyntacticFeaturizer.process'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_CountVectorsFeaturizer3' running 'CountVectorsFeaturizer.process'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_LanguageModelFeaturizer4' running 'LanguageModelFeaturizer.process'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_DIETClassifier5' running 'DIETClassifier.process'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_EntitySynonymMapper6' running 'EntitySynonymMapper.process'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_ResponseSelector7' running 'ResponseSelector.process'.
  2024-10-23 23:01:13 DEBUG    rasa.nlu.classifiers.diet_classifier  - There is no trained model for 'ResponseSelector': The component is either not trained or didn't receive enough training data.
  2024-10-23 23:01:13 DEBUG    rasa.nlu.selectors.response_selector  - Adding following selector key to message property: default
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_RegexMessageHandler' running 'RegexMessageHandler.process'.
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - [debug    ] processor.message.parse        parse_data_entities=[] parse_data_intent={'name': 'приветствие', 'confidence': 0.08105673640966415} parse_data_text=Привет!
  /home/pululun/yandex/rasa_env/lib64/python3.9/site-packages/rasa/shared/utils/io.py:99: UserWarning: Parsed an intent 'приветствие' which is not defined in the domain. Please make sure all intents are listed in the domain.
  More info at https://rasa.com/docs/rasa/domain
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - Logged UserUtterance - tracker now has 4 events.
  2024-10-23 23:01:13 DEBUG    rasa.core.actions.action  - Validating extracted slots:
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - [debug    ] processor.extract.slots        action_extract_slot=action_extract_slots len_extraction_events=0 rasa_events=[]
  2024-10-23 23:01:13 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2dbff6850>}, targets: ['select_prediction'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'rule_only_data_provider' running 'RuleOnlyDataProvider.provide'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_MemoizationPolicy0' running 'MemoizationPolicy.predict_action_probabilities'.
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.memoization  - [debug    ] memoization.predict.actions    tracker_states=[{}, {'user': {'intent': 'приветствие'}, 'prev_action': {'action_name': 'action_listen'}}]
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.memoization  - There is a memorised next action 'utter_greet'
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_RulePolicy1' running 'RulePolicy.predict_action_probabilities'.
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user text: Привет! | previous action name: action_listen
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.rule_policy  - There is no applicable rule.
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.rule_policy  - There is a rule for the next action 'utter_greet'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_TEDPolicy2' running 'TEDPolicy.predict_action_probabilities'.
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.ted_policy  - TED predicted 'action_listen' based on user intent.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'select_prediction' running 'DefaultPolicyPredictionEnsemble.combine_predictions_from_kwargs'.
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.ensemble  - Made prediction using user intent.
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.ensemble  - Added `DefinePrevUserUtteredFeaturization(False)` event.
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.ensemble  - Predicted next action using RulePolicy.
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - Predicted next action 'utter_greet' with confidence 1.00.
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[<rasa.shared.core.events.DefinePrevUserUtteredFeaturization object at 0x7fb2dbf0cdc0>]
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=utter_greet rasa_events=[BotUttered('Привет! Как я могу вам помочь?', {"elements": null, "quick_replies": null, "buttons": null, "attachment": null, "image": null, "custom": null}, {"utter_action": "utter_greet"}, 1729713673.9870975)]
  2024-10-23 23:01:13 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2dbff6850>}, targets: ['select_prediction'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'rule_only_data_provider' running 'RuleOnlyDataProvider.provide'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_MemoizationPolicy0' running 'MemoizationPolicy.predict_action_probabilities'.
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.memoization  - [debug    ] memoization.predict.actions    tracker_states=[{'user': {'intent': 'приветствие'}, 'prev_action': {'action_name': 'action_listen'}}, {'user': {'intent': 'приветствие'}, 'prev_action': {'action_name': 'utter_greet'}}]
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.memoization  - There is a memorised next action 'action_listen'
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_RulePolicy1' running 'RulePolicy.predict_action_probabilities'.
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  [state 2] user intent: приветствие | previous action name: utter_greet
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.rule_policy  - There is a rule for the next action 'action_listen'.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'run_TEDPolicy2' running 'TEDPolicy.predict_action_probabilities'.
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.ted_policy  - TED predicted 'action_listen' based on user intent.
  2024-10-23 23:01:13 DEBUG    rasa.engine.graph  - Node 'select_prediction' running 'DefaultPolicyPredictionEnsemble.combine_predictions_from_kwargs'.
  2024-10-23 23:01:13 DEBUG    rasa.core.policies.ensemble  - Predicted next action using RulePolicy.
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - Predicted next action 'action_listen' with confidence 1.00.
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[]
  2024-10-23 23:01:13 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=action_listen rasa_events=[]
  2024-10-23 23:01:13 DEBUG    rasa.core.tracker_store  - No event broker configured. Skipping streaming events.
  2024-10-23 23:01:13 DEBUG    rasa.core.lock_store  - Deleted lock for conversation 'name'.
  2024-10-23 23:02:17 DEBUG    rasa.core.lock_store  - Issuing ticket for conversation 'shreyas'.
  2024-10-23 23:02:17 DEBUG    rasa.core.lock_store  - Acquiring lock for conversation 'shreyas'.
  2024-10-23 23:02:17 DEBUG    rasa.core.lock_store  - Acquired lock for conversation 'shreyas'.
  2024-10-23 23:02:17 DEBUG    rasa.core.tracker_store  - Could not find tracker for conversation ID 'shreyas'.
  2024-10-23 23:02:17 DEBUG    rasa.core.tracker_store  - No event broker configured. Skipping streaming events.
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - Starting a new session for conversation ID 'shreyas'.
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[]
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=action_session_start rasa_events=[<rasa.shared.core.events.SessionStarted object at 0x7fb307e9e4f0>, ActionExecuted(action: action_listen, policy: None, confidence: None)]
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - [debug    ] processor.slots.log            slot_values=	session_started_metadata: None
  2024-10-23 23:02:17 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__message__': [<rasa.core.channels.channel.UserMessage object at 0x7fb2e0009a30>], '__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2e0011940>}, targets: ['run_RegexMessageHandler'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'nlu_message_converter' running 'NLUMessageConverter.convert_user_message'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_WhitespaceTokenizer0' running 'WhitespaceTokenizer.process'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_RegexFeaturizer1' running 'RegexFeaturizer.process'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_LexicalSyntacticFeaturizer2' running 'LexicalSyntacticFeaturizer.process'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_CountVectorsFeaturizer3' running 'CountVectorsFeaturizer.process'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_LanguageModelFeaturizer4' running 'LanguageModelFeaturizer.process'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_DIETClassifier5' running 'DIETClassifier.process'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_EntitySynonymMapper6' running 'EntitySynonymMapper.process'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_ResponseSelector7' running 'ResponseSelector.process'.
  2024-10-23 23:02:17 DEBUG    rasa.nlu.classifiers.diet_classifier  - There is no trained model for 'ResponseSelector': The component is either not trained or didn't receive enough training data.
  2024-10-23 23:02:17 DEBUG    rasa.nlu.selectors.response_selector  - Adding following selector key to message property: default
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_RegexMessageHandler' running 'RegexMessageHandler.process'.
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - [debug    ] processor.message.parse        parse_data_entities=[] parse_data_intent={'name': 'приветствие', 'confidence': 0.08105673640966415} parse_data_text=Привет!
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - Logged UserUtterance - tracker now has 4 events.
  2024-10-23 23:02:17 DEBUG    rasa.core.actions.action  - Validating extracted slots:
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - [debug    ] processor.extract.slots        action_extract_slot=action_extract_slots len_extraction_events=0 rasa_events=[]
  2024-10-23 23:02:17 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2e0011940>}, targets: ['select_prediction'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'rule_only_data_provider' running 'RuleOnlyDataProvider.provide'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_MemoizationPolicy0' running 'MemoizationPolicy.predict_action_probabilities'.
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.memoization  - [debug    ] memoization.predict.actions    tracker_states=[{}, {'user': {'intent': 'приветствие'}, 'prev_action': {'action_name': 'action_listen'}}]
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.memoization  - There is a memorised next action 'utter_greet'
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_RulePolicy1' running 'RulePolicy.predict_action_probabilities'.
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user text: Привет! | previous action name: action_listen
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.rule_policy  - There is no applicable rule.
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.rule_policy  - There is a rule for the next action 'utter_greet'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_TEDPolicy2' running 'TEDPolicy.predict_action_probabilities'.
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.ted_policy  - TED predicted 'action_listen' based on user intent.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'select_prediction' running 'DefaultPolicyPredictionEnsemble.combine_predictions_from_kwargs'.
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.ensemble  - Made prediction using user intent.
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.ensemble  - Added `DefinePrevUserUtteredFeaturization(False)` event.
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.ensemble  - Predicted next action using RulePolicy.
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - Predicted next action 'utter_greet' with confidence 1.00.
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[<rasa.shared.core.events.DefinePrevUserUtteredFeaturization object at 0x7fb307e9e9d0>]
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=utter_greet rasa_events=[BotUttered('Привет! Как я могу вам помочь?', {"elements": null, "quick_replies": null, "buttons": null, "attachment": null, "image": null, "custom": null}, {"utter_action": "utter_greet"}, 1729713737.756293)]
  2024-10-23 23:02:17 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2e0011940>}, targets: ['select_prediction'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'rule_only_data_provider' running 'RuleOnlyDataProvider.provide'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_MemoizationPolicy0' running 'MemoizationPolicy.predict_action_probabilities'.
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.memoization  - [debug    ] memoization.predict.actions    tracker_states=[{'user': {'intent': 'приветствие'}, 'prev_action': {'action_name': 'action_listen'}}, {'user': {'intent': 'приветствие'}, 'prev_action': {'action_name': 'utter_greet'}}]
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.memoization  - There is a memorised next action 'action_listen'
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_RulePolicy1' running 'RulePolicy.predict_action_probabilities'.
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  [state 2] user intent: приветствие | previous action name: utter_greet
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.rule_policy  - There is a rule for the next action 'action_listen'.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'run_TEDPolicy2' running 'TEDPolicy.predict_action_probabilities'.
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.ted_policy  - TED predicted 'action_listen' based on user intent.
  2024-10-23 23:02:17 DEBUG    rasa.engine.graph  - Node 'select_prediction' running 'DefaultPolicyPredictionEnsemble.combine_predictions_from_kwargs'.
  2024-10-23 23:02:17 DEBUG    rasa.core.policies.ensemble  - Predicted next action using RulePolicy.
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - Predicted next action 'action_listen' with confidence 1.00.
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[]
  2024-10-23 23:02:17 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=action_listen rasa_events=[]
  2024-10-23 23:02:17 DEBUG    rasa.core.tracker_store  - No event broker configured. Skipping streaming events.
  2024-10-23 23:02:17 DEBUG    rasa.core.lock_store  - Deleted lock for conversation 'shreyas'.
  2024-10-23 23:03:52 DEBUG    rasa.core.lock_store  - Issuing ticket for conversation 'name'.
  2024-10-23 23:03:52 DEBUG    rasa.core.lock_store  - Acquiring lock for conversation 'name'.
  2024-10-23 23:03:52 DEBUG    rasa.core.lock_store  - Acquired lock for conversation 'name'.
  2024-10-23 23:03:52 DEBUG    rasa.core.tracker_store  - Recreating tracker for id 'name'
  2024-10-23 23:03:52 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__message__': [<rasa.core.channels.channel.UserMessage object at 0x7fb2e00092b0>], '__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2dbf0cdc0>}, targets: ['run_RegexMessageHandler'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'nlu_message_converter' running 'NLUMessageConverter.convert_user_message'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_WhitespaceTokenizer0' running 'WhitespaceTokenizer.process'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_RegexFeaturizer1' running 'RegexFeaturizer.process'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_LexicalSyntacticFeaturizer2' running 'LexicalSyntacticFeaturizer.process'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_CountVectorsFeaturizer3' running 'CountVectorsFeaturizer.process'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_LanguageModelFeaturizer4' running 'LanguageModelFeaturizer.process'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_DIETClassifier5' running 'DIETClassifier.process'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_EntitySynonymMapper6' running 'EntitySynonymMapper.process'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_ResponseSelector7' running 'ResponseSelector.process'.
  2024-10-23 23:03:52 DEBUG    rasa.nlu.classifiers.diet_classifier  - There is no trained model for 'ResponseSelector': The component is either not trained or didn't receive enough training data.
  2024-10-23 23:03:52 DEBUG    rasa.nlu.selectors.response_selector  - Adding following selector key to message property: default
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_RegexMessageHandler' running 'RegexMessageHandler.process'.
  2024-10-23 23:03:52 DEBUG    rasa.core.processor  - [debug    ] processor.message.parse        parse_data_entities=[] parse_data_intent={'name': 'микросервисы', 'confidence': 0.06321300566196442} parse_data_text=Что такое микросервисная архитектура?
  /home/pululun/yandex/rasa_env/lib64/python3.9/site-packages/rasa/shared/utils/io.py:99: UserWarning: Parsed an intent 'микросервисы' which is not defined in the domain. Please make sure all intents are listed in the domain.
  More info at https://rasa.com/docs/rasa/domain
  2024-10-23 23:03:52 DEBUG    rasa.core.processor  - Logged UserUtterance - tracker now has 9 events.
  2024-10-23 23:03:52 DEBUG    rasa.core.actions.action  - Validating extracted slots:
  2024-10-23 23:03:52 DEBUG    rasa.core.processor  - [debug    ] processor.extract.slots        action_extract_slot=action_extract_slots len_extraction_events=0 rasa_events=[]
  2024-10-23 23:03:52 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2dbf0cdc0>}, targets: ['select_prediction'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'rule_only_data_provider' running 'RuleOnlyDataProvider.provide'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_MemoizationPolicy0' running 'MemoizationPolicy.predict_action_probabilities'.
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.memoization  - [debug    ] memoization.predict.actions    tracker_states=[{'user': {'intent': 'приветствие'}, 'prev_action': {'action_name': 'utter_greet'}}, {'user': {'intent': 'микросервисы'}, 'prev_action': {'action_name': 'action_listen'}}]
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.memoization  - There is no memorised next action
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_RulePolicy1' running 'RulePolicy.predict_action_probabilities'.
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  [state 2] user intent: приветствие | previous action name: utter_greet
  [state 3] user text: Что такое микросервисная архитектура? | previous action name: action_listen
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.rule_policy  - There is no applicable rule.
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  [state 2] user intent: приветствие | previous action name: utter_greet
  [state 3] user intent: микросервисы | previous action name: action_listen
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.rule_policy  - There is a rule for the next action 'utter_microservices'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_TEDPolicy2' running 'TEDPolicy.predict_action_probabilities'.
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.ted_policy  - TED predicted 'utter_documentation' based on user intent.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'select_prediction' running 'DefaultPolicyPredictionEnsemble.combine_predictions_from_kwargs'.
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.ensemble  - Made prediction using user intent.
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.ensemble  - Added `DefinePrevUserUtteredFeaturization(False)` event.
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.ensemble  - Predicted next action using RulePolicy.
  2024-10-23 23:03:52 DEBUG    rasa.core.processor  - Predicted next action 'utter_microservices' with confidence 1.00.
  2024-10-23 23:03:52 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[<rasa.shared.core.events.DefinePrevUserUtteredFeaturization object at 0x7fb307e9e9d0>]
  2024-10-23 23:03:52 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=utter_microservices rasa_events=[BotUttered('Микросервисная архитектура — это подход к разработке, когда приложение разделяется на независимые сервисы, взаимодействующие между собой.', {"elements": null, "quick_replies": null, "buttons": null, "attachment": null, "image": null, "custom": null}, {"utter_action": "utter_microservices"}, 1729713832.8208013)]
  2024-10-23 23:03:52 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2dbf0cdc0>}, targets: ['select_prediction'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'rule_only_data_provider' running 'RuleOnlyDataProvider.provide'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_MemoizationPolicy0' running 'MemoizationPolicy.predict_action_probabilities'.
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.memoization  - [debug    ] memoization.predict.actions    tracker_states=[{'user': {'intent': 'микросервисы'}, 'prev_action': {'action_name': 'action_listen'}}, {'user': {'intent': 'микросервисы'}, 'prev_action': {'action_name': 'utter_microservices'}}]
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.memoization  - There is a memorised next action 'action_listen'
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_RulePolicy1' running 'RulePolicy.predict_action_probabilities'.
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  [state 2] user intent: приветствие | previous action name: utter_greet
  [state 3] user intent: микросервисы | previous action name: action_listen
  [state 4] user intent: микросервисы | previous action name: utter_microservices
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.rule_policy  - There is a rule for the next action 'action_listen'.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'run_TEDPolicy2' running 'TEDPolicy.predict_action_probabilities'.
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.ted_policy  - TED predicted 'action_listen' based on user intent.
  2024-10-23 23:03:52 DEBUG    rasa.engine.graph  - Node 'select_prediction' running 'DefaultPolicyPredictionEnsemble.combine_predictions_from_kwargs'.
  2024-10-23 23:03:52 DEBUG    rasa.core.policies.ensemble  - Predicted next action using RulePolicy.
  2024-10-23 23:03:52 DEBUG    rasa.core.processor  - Predicted next action 'action_listen' with confidence 1.00.
  2024-10-23 23:03:52 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[]
  2024-10-23 23:03:52 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=action_listen rasa_events=[]
  2024-10-23 23:03:52 DEBUG    rasa.core.tracker_store  - No event broker configured. Skipping streaming events.
  2024-10-23 23:03:52 DEBUG    rasa.core.lock_store  - Deleted lock for conversation 'name'.
  2024-10-23 23:04:45 DEBUG    rasa.core.lock_store  - Issuing ticket for conversation 'name'.
  2024-10-23 23:04:45 DEBUG    rasa.core.lock_store  - Acquiring lock for conversation 'name'.
  2024-10-23 23:04:45 DEBUG    rasa.core.lock_store  - Acquired lock for conversation 'name'.
  2024-10-23 23:04:45 DEBUG    rasa.core.tracker_store  - Recreating tracker for id 'name'
  2024-10-23 23:04:45 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__message__': [<rasa.core.channels.channel.UserMessage object at 0x7fb2e3f43c70>], '__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2e00095e0>}, targets: ['run_RegexMessageHandler'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'nlu_message_converter' running 'NLUMessageConverter.convert_user_message'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_WhitespaceTokenizer0' running 'WhitespaceTokenizer.process'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_RegexFeaturizer1' running 'RegexFeaturizer.process'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_LexicalSyntacticFeaturizer2' running 'LexicalSyntacticFeaturizer.process'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_CountVectorsFeaturizer3' running 'CountVectorsFeaturizer.process'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_LanguageModelFeaturizer4' running 'LanguageModelFeaturizer.process'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_DIETClassifier5' running 'DIETClassifier.process'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_EntitySynonymMapper6' running 'EntitySynonymMapper.process'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_ResponseSelector7' running 'ResponseSelector.process'.
  2024-10-23 23:04:45 DEBUG    rasa.nlu.classifiers.diet_classifier  - There is no trained model for 'ResponseSelector': The component is either not trained or didn't receive enough training data.
  2024-10-23 23:04:45 DEBUG    rasa.nlu.selectors.response_selector  - Adding following selector key to message property: default
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_RegexMessageHandler' running 'RegexMessageHandler.process'.
  2024-10-23 23:04:45 DEBUG    rasa.core.processor  - [debug    ] processor.message.parse        parse_data_entities=[] parse_data_intent={'name': 'микросервисы', 'confidence': 0.06873173266649246} parse_data_text=Как реализовать микросервисную архитектуру?
  2024-10-23 23:04:45 DEBUG    rasa.core.processor  - Logged UserUtterance - tracker now has 14 events.
  2024-10-23 23:04:45 DEBUG    rasa.core.actions.action  - Validating extracted slots:
  2024-10-23 23:04:45 DEBUG    rasa.core.processor  - [debug    ] processor.extract.slots        action_extract_slot=action_extract_slots len_extraction_events=0 rasa_events=[]
  2024-10-23 23:04:45 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2e00095e0>}, targets: ['select_prediction'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'rule_only_data_provider' running 'RuleOnlyDataProvider.provide'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_MemoizationPolicy0' running 'MemoizationPolicy.predict_action_probabilities'.
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.memoization  - [debug    ] memoization.predict.actions    tracker_states=[{'user': {'intent': 'микросервисы'}, 'prev_action': {'action_name': 'utter_microservices'}}, {'user': {'intent': 'микросервисы'}, 'prev_action': {'action_name': 'action_listen'}}]
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.memoization  - There is no memorised next action
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_RulePolicy1' running 'RulePolicy.predict_action_probabilities'.
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  [state 2] user intent: приветствие | previous action name: utter_greet
  [state 3] user intent: микросервисы | previous action name: action_listen
  [state 4] user intent: микросервисы | previous action name: utter_microservices
  [state 5] user text: Как реализовать микросервисную архитектуру? | previous action name: action_listen
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.rule_policy  - There is no applicable rule.
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  [state 2] user intent: приветствие | previous action name: utter_greet
  [state 3] user intent: микросервисы | previous action name: action_listen
  [state 4] user intent: микросервисы | previous action name: utter_microservices
  [state 5] user intent: микросервисы | previous action name: action_listen
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.rule_policy  - There is a rule for the next action 'utter_microservices'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_TEDPolicy2' running 'TEDPolicy.predict_action_probabilities'.
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.ted_policy  - TED predicted 'utter_documentation' based on user intent.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'select_prediction' running 'DefaultPolicyPredictionEnsemble.combine_predictions_from_kwargs'.
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.ensemble  - Made prediction using user intent.
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.ensemble  - Added `DefinePrevUserUtteredFeaturization(False)` event.
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.ensemble  - Predicted next action using RulePolicy.
  2024-10-23 23:04:45 DEBUG    rasa.core.processor  - Predicted next action 'utter_microservices' with confidence 1.00.
  2024-10-23 23:04:45 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[<rasa.shared.core.events.DefinePrevUserUtteredFeaturization object at 0x7fb2e00092b0>]
  2024-10-23 23:04:45 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=utter_microservices rasa_events=[BotUttered('Микросервисная архитектура — это подход к разработке, когда приложение разделяется на независимые сервисы, взаимодействующие между собой.', {"elements": null, "quick_replies": null, "buttons": null, "attachment": null, "image": null, "custom": null}, {"utter_action": "utter_microservices"}, 1729713885.1926856)]
  2024-10-23 23:04:45 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2e00095e0>}, targets: ['select_prediction'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'rule_only_data_provider' running 'RuleOnlyDataProvider.provide'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_MemoizationPolicy0' running 'MemoizationPolicy.predict_action_probabilities'.
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.memoization  - [debug    ] memoization.predict.actions    tracker_states=[{'user': {'intent': 'микросервисы'}, 'prev_action': {'action_name': 'action_listen'}}, {'user': {'intent': 'микросервисы'}, 'prev_action': {'action_name': 'utter_microservices'}}]
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.memoization  - There is a memorised next action 'action_listen'
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_RulePolicy1' running 'RulePolicy.predict_action_probabilities'.
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  [state 2] user intent: приветствие | previous action name: utter_greet
  [state 3] user intent: микросервисы | previous action name: action_listen
  [state 4] user intent: микросервисы | previous action name: utter_microservices
  [state 5] user intent: микросервисы | previous action name: action_listen
  [state 6] user intent: микросервисы | previous action name: utter_microservices
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.rule_policy  - There is a rule for the next action 'action_listen'.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'run_TEDPolicy2' running 'TEDPolicy.predict_action_probabilities'.
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.ted_policy  - TED predicted 'action_listen' based on user intent.
  2024-10-23 23:04:45 DEBUG    rasa.engine.graph  - Node 'select_prediction' running 'DefaultPolicyPredictionEnsemble.combine_predictions_from_kwargs'.
  2024-10-23 23:04:45 DEBUG    rasa.core.policies.ensemble  - Predicted next action using RulePolicy.
  2024-10-23 23:04:45 DEBUG    rasa.core.processor  - Predicted next action 'action_listen' with confidence 1.00.
  2024-10-23 23:04:45 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[]
  2024-10-23 23:04:45 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=action_listen rasa_events=[]
  2024-10-23 23:04:45 DEBUG    rasa.core.tracker_store  - No event broker configured. Skipping streaming events.
  2024-10-23 23:04:45 DEBUG    rasa.core.lock_store  - Deleted lock for conversation 'name'.
  2024-10-23 23:05:24 DEBUG    rasa.core.lock_store  - Issuing ticket for conversation 'name'.
  2024-10-23 23:05:24 DEBUG    rasa.core.lock_store  - Acquiring lock for conversation 'name'.
  2024-10-23 23:05:24 DEBUG    rasa.core.lock_store  - Acquired lock for conversation 'name'.
  2024-10-23 23:05:24 DEBUG    rasa.core.tracker_store  - Recreating tracker for id 'name'
  2024-10-23 23:05:24 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__message__': [<rasa.core.channels.channel.UserMessage object at 0x7fb2e35f2f70>], '__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2e2bdefa0>}, targets: ['run_RegexMessageHandler'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'nlu_message_converter' running 'NLUMessageConverter.convert_user_message'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_WhitespaceTokenizer0' running 'WhitespaceTokenizer.process'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_RegexFeaturizer1' running 'RegexFeaturizer.process'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_LexicalSyntacticFeaturizer2' running 'LexicalSyntacticFeaturizer.process'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_CountVectorsFeaturizer3' running 'CountVectorsFeaturizer.process'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_LanguageModelFeaturizer4' running 'LanguageModelFeaturizer.process'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_DIETClassifier5' running 'DIETClassifier.process'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_EntitySynonymMapper6' running 'EntitySynonymMapper.process'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_ResponseSelector7' running 'ResponseSelector.process'.
  2024-10-23 23:05:24 DEBUG    rasa.nlu.classifiers.diet_classifier  - There is no trained model for 'ResponseSelector': The component is either not trained or didn't receive enough training data.
  2024-10-23 23:05:24 DEBUG    rasa.nlu.selectors.response_selector  - Adding following selector key to message property: default
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_RegexMessageHandler' running 'RegexMessageHandler.process'.
  2024-10-23 23:05:24 DEBUG    rasa.core.processor  - [debug    ] processor.message.parse        parse_data_entities=[] parse_data_intent={'name': 'запрос_помощи', 'confidence': 0.06625044345855713} parse_data_text=Как её реализовать?
  /home/pululun/yandex/rasa_env/lib64/python3.9/site-packages/rasa/shared/utils/io.py:99: UserWarning: Parsed an intent 'запрос_помощи' which is not defined in the domain. Please make sure all intents are listed in the domain.
  More info at https://rasa.com/docs/rasa/domain
  2024-10-23 23:05:24 DEBUG    rasa.core.processor  - Logged UserUtterance - tracker now has 19 events.
  2024-10-23 23:05:24 DEBUG    rasa.core.actions.action  - Validating extracted slots:
  2024-10-23 23:05:24 DEBUG    rasa.core.processor  - [debug    ] processor.extract.slots        action_extract_slot=action_extract_slots len_extraction_events=0 rasa_events=[]
  2024-10-23 23:05:24 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2e2bdefa0>}, targets: ['select_prediction'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'rule_only_data_provider' running 'RuleOnlyDataProvider.provide'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_MemoizationPolicy0' running 'MemoizationPolicy.predict_action_probabilities'.
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.memoization  - [debug    ] memoization.predict.actions    tracker_states=[{'user': {'intent': 'микросервисы'}, 'prev_action': {'action_name': 'utter_microservices'}}, {'user': {'intent': 'запрос_помощи'}, 'prev_action': {'action_name': 'action_listen'}}]
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.memoization  - There is no memorised next action
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_RulePolicy1' running 'RulePolicy.predict_action_probabilities'.
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  [state 2] user intent: приветствие | previous action name: utter_greet
  [state 3] user intent: микросервисы | previous action name: action_listen
  [state 4] user intent: микросервисы | previous action name: utter_microservices
  [state 5] user intent: микросервисы | previous action name: action_listen
  [state 6] user intent: микросервисы | previous action name: utter_microservices
  [state 7] user text: Как её реализовать? | previous action name: action_listen
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.rule_policy  - There is no applicable rule.
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  [state 2] user intent: приветствие | previous action name: utter_greet
  [state 3] user intent: микросервисы | previous action name: action_listen
  [state 4] user intent: микросервисы | previous action name: utter_microservices
  [state 5] user intent: микросервисы | previous action name: action_listen
  [state 6] user intent: микросервисы | previous action name: utter_microservices
  [state 7] user intent: запрос_помощи | previous action name: action_listen
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.rule_policy  - There is a rule for the next action 'utter_help'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_TEDPolicy2' running 'TEDPolicy.predict_action_probabilities'.
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.ted_policy  - TED predicted 'utter_documentation' based on user intent.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'select_prediction' running 'DefaultPolicyPredictionEnsemble.combine_predictions_from_kwargs'.
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.ensemble  - Made prediction using user intent.
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.ensemble  - Added `DefinePrevUserUtteredFeaturization(False)` event.
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.ensemble  - Predicted next action using RulePolicy.
  2024-10-23 23:05:24 DEBUG    rasa.core.processor  - Predicted next action 'utter_help' with confidence 1.00.
  2024-10-23 23:05:24 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[<rasa.shared.core.events.DefinePrevUserUtteredFeaturization object at 0x7fb307e9e9d0>]
  2024-10-23 23:05:24 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=utter_help rasa_events=[BotUttered('Чем я могу вам помочь?', {"elements": null, "quick_replies": null, "buttons": null, "attachment": null, "image": null, "custom": null}, {"utter_action": "utter_help"}, 1729713924.626922)]
  2024-10-23 23:05:24 DEBUG    rasa.engine.runner.dask  - Running graph with inputs: {'__tracker__': <rasa.shared.core.trackers.DialogueStateTracker object at 0x7fb2e2bdefa0>}, targets: ['select_prediction'] and ExecutionContext(model_id='f256418281674c3f86c5095106307c68', should_add_diagnostic_data=False, is_finetuning=False, node_name=None).
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'rule_only_data_provider' running 'RuleOnlyDataProvider.provide'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'domain_provider' running 'DomainProvider.provide_inference'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_MemoizationPolicy0' running 'MemoizationPolicy.predict_action_probabilities'.
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.memoization  - [debug    ] memoization.predict.actions    tracker_states=[{'user': {'intent': 'запрос_помощи'}, 'prev_action': {'action_name': 'action_listen'}}, {'user': {'intent': 'запрос_помощи'}, 'prev_action': {'action_name': 'utter_help'}}]
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.memoization  - There is a memorised next action 'action_listen'
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_RulePolicy1' running 'RulePolicy.predict_action_probabilities'.
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.rule_policy  - [debug    ] rule_policy.actions.find       current_states=
  [state 1] user intent: приветствие | previous action name: action_listen
  [state 2] user intent: приветствие | previous action name: utter_greet
  [state 3] user intent: микросервисы | previous action name: action_listen
  [state 4] user intent: микросервисы | previous action name: utter_microservices
  [state 5] user intent: микросервисы | previous action name: action_listen
  [state 6] user intent: микросервисы | previous action name: utter_microservices
  [state 7] user intent: запрос_помощи | previous action name: action_listen
  [state 8] user intent: запрос_помощи | previous action name: utter_help
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.rule_policy  - There is a rule for the next action 'action_listen'.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'run_TEDPolicy2' running 'TEDPolicy.predict_action_probabilities'.
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.ted_policy  - TED predicted 'action_listen' based on user intent.
  2024-10-23 23:05:24 DEBUG    rasa.engine.graph  - Node 'select_prediction' running 'DefaultPolicyPredictionEnsemble.combine_predictions_from_kwargs'.
  2024-10-23 23:05:24 DEBUG    rasa.core.policies.ensemble  - Predicted next action using RulePolicy.
  2024-10-23 23:05:24 DEBUG    rasa.core.processor  - Predicted next action 'action_listen' with confidence 1.00.
  2024-10-23 23:05:24 DEBUG    rasa.core.processor  - [debug    ] processor.actions.policy_prediction prediction_events=[]
  2024-10-23 23:05:24 DEBUG    rasa.core.processor  - [debug    ] processor.actions.log          action_name=action_listen rasa_events=[]
  2024-10-23 23:05:24 DEBUG    rasa.core.tracker_store  - No event broker configured. Skipping streaming events.
  2024-10-23 23:05:24 DEBUG    rasa.core.lock_store  - Deleted lock for conversation 'name'.