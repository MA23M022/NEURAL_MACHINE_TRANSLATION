

def train_epoch(dataloader, model, optimizer, criterion):
    iterations = 0
    total_loss = 0
    print_loss_total = 0

    for data in dataloader:
        if iterations % 100 == 0 and iterations != 0:
            print_loss_avg = print_loss_total / 100.0
            print_loss_total = 0
            print(f"At the {iterations}-th iteration, the training loss is {print_loss_avg}")

        input_tensor, target_tensor = data

        optimizer.zero_grad()
        decoder_outputs, _, _ = model(input_tensor, target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print_loss_total += loss.item()
        iterations += 1

    return total_loss / len(dataloader)
