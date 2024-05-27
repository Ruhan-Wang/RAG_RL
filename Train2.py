import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from datasets import load_dataset
from ralm.file_utils import print_args
from ralm.model_utils import load_model_and_tokenizer


# Prediction of Rewards
class BanditNetwork(nn.Module):
    def __init__(self):
        super(BanditNetwork, self).__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)  # Output layer for predicted reward
        self.fc3 = nn.Linear(128, 1)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        # x = self.softmax(x)
        return x


# Action Selection
def epsilon_greedy_action(action_values, epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(len(action_values))
    else:
        return np.argmax(action_values, axis=0)


def get_predict_sentence(input_ids, outputs, trg_len, max_length):
    if trg_len < max_length:
        lm_logits = outputs.logits[..., -trg_len - 1:-1, :]
    else:
        lm_logits = outputs.logits[..., :-1, :]
    probs = torch.softmax(lm_logits, dim=-1)
    predicted_token_ids = torch.argmax(probs, dim=-1)
    input_ids[:, -trg_len:] = predicted_token_ids
    return input_ids


def get_feedback(predict_sentence, target_sentence):
    criterion = nn.MSELoss()
    feedback = criterion(predict_sentence, target_sentence)
    return -feedback


def claculate_reward(model, input_ids, target_ids, trg_len, max_length):
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        if trg_len < max_length:
            lm_logits = outputs.logits[..., -trg_len - 1:-1, :]
            labels = target_ids[..., -trg_len:]
        else:
            lm_logits = outputs.logits[..., :-1, :]
            labels = target_ids[..., 1:]
        lm_logits = lm_logits.to(torch.float32)
        loss_fct = CrossEntropyLoss(reduction="none")
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        ppl = torch.exp(torch.sum(loss) / len(loss))
    return ppl


def online_train_bandit(llm_model, bandit_model, learning_rate, epsilon, tokenizer, dataset, device,
                        max_length, stride=4, normalization_level="word", retrieval_dataset=None,
                        retrieval_max_length=256):
    encodings = tokenizer(dataset, add_special_tokens=False, return_tensors="pt")
    for param in llm_model.parameters():
        param.requires_grad = False
    print("Max context length:", max_length)
    # Number of tokens in dataset
    dataset_len = encodings.input_ids.size(1)
    print("Dataset length:", dataset_len)
    if normalization_level == "word":
        counter = dataset.count(" ")
    elif normalization_level == "token":
        counter = dataset_len
    else:
        raise ValueError(f"Unknown normalization_level: '{normalization_level}'")

    print("Normalization factor (num tokens/words..):", counter)
    prev_end_loc = 0
    idx = 0
    bandit_model.cuda().train()
    optimizer = optim.Adam(bandit_model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    loss_ls = []
    ppl_ls = []
    for begin_loc in tqdm(range(0, dataset_len, stride)):
        end_loc = min(begin_loc + max_length, dataset_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        if idx > 0 and retrieval_dataset is not None and len(retrieval_dataset[idx]["retrieved_docs"]) > 0:
            retrieved_item = retrieval_dataset[idx]
            assert retrieved_item["begin_location"] == prev_end_loc
            assert retrieved_item["end_location"] == end_loc
            original_input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            num_docs = len(retrieved_item["retrieved_docs"])
            input_ids = original_input_ids.repeat(num_docs, 1)
            target_ids = original_input_ids.clone()
            for doc_id in range(num_docs):
                retrieved_example = retrieved_item["retrieved_docs"][doc_id]

                doc_title = retrieved_example["title"] if "title" in retrieved_example else None
                doc_text = retrieved_example["text"]
                if doc_title:
                    doc_text = doc_title + "\n" + doc_text
                encoded_retrieved_text = tokenizer.encode(doc_text, max_length=retrieval_max_length, truncation=True)

                input_ids[doc_id, :len(encoded_retrieved_text)] = torch.tensor(encoded_retrieved_text, device=device)
            if end_loc == dataset_len:
                break
            retrieval_input = input_ids[0, :]
            target_ids[:, :-trg_len] = -100
            ppl = claculate_reward(llm_model, retrieval_input, target_ids, trg_len, max_length)
            ppl_ls.append(ppl.item())
        prev_end_loc = end_loc
        idx += 1
        if end_loc == dataset_len:
            break
    mean_ppl = np.mean(ppl_ls)
    return mean_ppl, ppl_ls

    #         predicted_all_actions = bandit_model(input_ids.float())
    #         action = epsilon_greedy_action(predicted_all_actions.cpu().detach().numpy(), epsilon)
    #         retrieval_input = input_ids[action, :]
    #         target_ids[:, :-trg_len] = -100
    #         ppl = claculate_reward(llm_model, retrieval_input, target_ids, trg_len, max_length)
    #         choose_output = llm_model(retrieval_input, labels=target_ids)
    #         feedback_input = get_predict_sentence(original_input_ids, choose_output, trg_len, max_length)
    #         feedback_target = original_input_ids
    #         feedback = get_feedback(feedback_input.float(), feedback_target.float())
    #         loss = loss_fn(predicted_all_actions[action], feedback)
    #         ppl_ls.append(ppl.item())
    #         loss_ls.append(loss.item())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     prev_end_loc = end_loc
    #     idx += 1
    #     if end_loc == dataset_len:
    #         break
    # mean_loss = np.mean(loss_ls)
    # mean_ppl = np.mean(ppl_ls)
    # return loss_ls, ppl_ls, mean_loss, mean_ppl


def main(args):
    llm_model, tokenizer, config, device = load_model_and_tokenizer(
        args.model_name, model_parallelism=args.model_parallelism, cache_dir=args.cache_dir,
        auth_token=args.auth_token
    )

    # Model context size (e.g., 1024 for GPT-2)
    max_length = args.max_length
    model_max_length = config.n_positions if hasattr(config,
                                                     "n_positions") else config.max_position_embeddings
    if max_length is None or max_length > model_max_length:
        max_length = model_max_length

    if args.load_from == "hf":
        dataset = load_dataset(args.dataset_path, args.dataset_name, split=args.dataset_split)
        dataset = "".join([x["text"] if x["text"] else " \n" for x in dataset])
    else:
        with open(args.dataset_path, "r") as f:
            dataset = f.read()

    transformers.logging.set_verbosity_error()
    retrieval_dataset = None
    if args.retrieved_file is not None:
        with open(args.retrieved_file, "r") as f:
            retrieval_dataset = json.load(f)
    bandit_model = BanditNetwork().to(device)
    mean_ppl, ppl_ls = online_train_bandit(llm_model, bandit_model, learning_rate=1e-4,
                                           epsilon=0.1,
                                           tokenizer=tokenizer,
                                           dataset=dataset,
                                           device=device, max_length=max_length, stride=4,
                                           normalization_level="word",
                                           retrieval_dataset=retrieval_dataset,
                                           retrieval_max_length=model_max_length)
    x = np.arange(len(ppl_ls))
    plt.figure(figsize=(8, 6))
    plt.plot(x, ppl_ls, color='red')
    plt.legend()
    plt.xlabel('token_batch', fontsize=20)
    plt.ylabel('PPL', fontsize=20)
    plt.title("Original_PPL", fontsize=20)
    plt.grid(True)
    plt.savefig("/home/ruhwang/Project/RAG_RL/Figure/Original_PPL.png")
    print("Mean PPL:{}".format(mean_ppl))

    # bandit_model = BanditNetwork().to(device)
    # iter_ppl = []
    # iter_loss = []
    # for i in range(10):
    #     loss_ls, ppl_ls, mean_loss, mean_ppl = online_train_bandit(llm_model, bandit_model, learning_rate=1e-4,
    #                                                                epsilon=0.1,
    #                                                                tokenizer=tokenizer,
    #                                                                dataset=dataset,
    #                                                                device=device, max_length=max_length, stride=4,
    #                                                                normalization_level="word",
    #                                                                retrieval_dataset=retrieval_dataset,
    #                                                                retrieval_max_length=model_max_length)
    #     x = np.arange(len(loss_ls))
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(x, loss_ls, color='blue')
    #     plt.legend()
    #     plt.xlabel('token_batch', fontsize=20)
    #     plt.ylabel('loss', fontsize=20)
    #     plt.title('Iter_' + str(i) + "_Loss", fontsize=20)
    #     plt.grid(True)
    #     plt.savefig("/home/ruhwang/Project/RAG_RL/Figure/" + 'Iter_' + str(i) + "_Loss.png")
    #
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(x, ppl_ls, color='red')
    #     plt.legend()
    #     plt.xlabel('token_batch', fontsize=20)
    #     plt.ylabel('PPL', fontsize=20)
    #     plt.title('Iter_' + str(i) + "_PPL", fontsize=20)
    #     plt.grid(True)
    #     plt.savefig("/home/ruhwang/Project/RAG_RL/Figure/" + 'Iter_' + str(i) + "_PPL.png")
    #
    #     iter_ppl.append(mean_ppl)
    #     iter_loss.append(mean_loss)
    #     print("Mean_ppl:{:.4f}     Mean_Loss:{:.4f}".format(mean_ppl, mean_loss))
    # return iter_ppl, iter_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--model_parallelism", action="store_true")
    parser.add_argument("--auth_token", type=str, default=None)

    # Dataset params
    parser.add_argument("--load_from", type=str, choices=["hf", "file"], default="hf")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--normalization_level", choices=["word", "token"], default="word")

    # retrieval params
    parser.add_argument("--retrieved_file", type=str, default=None)
    parser.add_argument("--retrieved_max_length", type=int, default=256)

    args = parser.parse_args()

    # iter_ppl, iter_loss = main(args)
    main(args)
    # x = np.arange(len(iter_loss))
    # plt.figure(figsize=(8, 6))
    # plt.plot(x, iter_loss, color='blue')
    # plt.legend()
    # plt.xlabel('iteration', fontsize=20)
    # plt.ylabel('iter_loss', fontsize=20)
    # plt.title("All_Loss", fontsize=20)
    # plt.grid(True)
    # plt.savefig("/home/ruhwang/Project/RAG_RL/Figure/All_Loss.png")
    #
    # plt.figure(figsize=(8, 6))
    # plt.plot(x, iter_ppl, color='red')
    # plt.legend()
    # plt.xlabel('iteration', fontsize=20)
    # plt.ylabel('iter_PPL', fontsize=20)
    # plt.title("All_PPL", fontsize=20)
    # plt.grid(True)
    # plt.savefig("/home/ruhwang/Project/RAG_RL/Figure/All_PPL.png")
