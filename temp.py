def load_data(data_split, batch_size=32, img_size=224, augment=False, shuffle=True):
    assert data_split in ['train', 'test', 'valid']
    img_path = 'data2/' + data_split


    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    if augment:
        transform = transforms.Compose([
                                        transforms.RandomResizedCrop(img_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(10),
                                        transforms.ToTensor(),
                                        normalize])
    else:
        transform = transforms.Compose([transforms.Resize(img_size),
                                        transforms.CenterCrop(img_size),
                                        transforms.ToTensor(),
                                        normalize])

    dataset = MyDataClass(img_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    # Estimate the mean, variance so we can normalize. This is only for one batch right now, though, so it could
    # likely be improved
    def mean_std(loader):
        images, _ = next(iter(loader))
        # shape of images = [b,c,w,h]
        mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])
        return mean, std

    mean, std = mean_std(dataloader)
    print("mean and std: \n", mean, std)
    return dataloader
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
