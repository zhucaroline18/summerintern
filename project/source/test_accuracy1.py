def test_model_accuracy():
    test_dataset = ImageDataset(test_src_image_folder, test_label_image_folder)
    # test_dataset = ImageDataset(train_src_image_folder, train_label_image_folder)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    model = torch.load(os.path.join(current_location, "saved-deeplab.pth"))
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        index = 0
        target_index = None
        accuracy_list = []

        for image, label in test_loader:
            output = model(image)
            target = label.float()
            loss = criterion(output, target)
            print(f"loss={loss}")

            output_raw = output.squeeze(0)

            # the index who has the max value. As we only have two classes, it's either 0 or 1
            output_label = output_raw.argmax(dim=0)

            image_data = image.squeeze(0)[0]
            label_data = label.squeeze(0)[1]

            if index == target_index:
                print(label_data - output_label)
                print(torch.abs(label_data - output_label).sum())

                # draw the images
                fig, axarr = plt.subplots(3)
                # show the image
                axarr[0].imshow(image_data, cmap = "gray")
                # expected label
                axarr[1].imshow(label_data, cmap = "gray")
                # defect label
                axarr[2].imshow(output_label, cmap = "gray")
                plt.show()

            # As we only have 0 and 1, the simple way is to substract two matrix
            # then get the sum of the absoluate value. Those are the pixels have difference
            accuracy = 1.0 - torch.abs(label_data - output_label).sum().item() / (image_height * image_width)            
            accuracy_list.append(accuracy)

            index = index + 1

        accuracy_list_np = np.array(accuracy_list)
        print(f"Accuracy: min={accuracy_list_np.min()}, max={accuracy_list_np.max()}, average={np.average(accuracy_list_np)}")
