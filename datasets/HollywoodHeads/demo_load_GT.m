%% setup
filePath = fileparts( mfilename('fullpath') );
addpath( fullfile(filePath, 'code' ) );

%% Init
img_folder = fullfile( filePath, 'JPEGImages' );
ann_folder = fullfile( filePath, 'Annotations' );
split_folder = fullfile( filePath, 'Splits' );

split_name = 'val'; % 'train', 'val' or 'test'
%% visualize GT
%read list of *split* image
list_file = readLines(fullfile(split_folder, [split_name '.txt']));

stop = false;
while ~stop
    fig = figure;
    
    idx = randi(length(list_file));
    
    filename = list_file{idx};
    
    im_path = fullfile(img_folder, [filename '.jpeg']);
    ann_path = fullfile(ann_folder, [filename '.xml']);
    
    rec=VOCreadrecxml(ann_path);
    
    image(imread(im_path)); hold on;
    for i=1:length(rec.objects)
        if rec.objects(i).difficult
            c = 'r'; % difficult annotation is drawn in red
        else
            c = 'y'; % normal annotation is drawn in yellow
        end
        
        bb = rec.objects(i).bbox;
        rectangle('position', [bb(:,1) bb(:,2) bb(:,3)-bb(:,1)+1 bb(:,4)-bb(:,2)+1], 'edgecolor', c, 'linewidth', 3);
    end
    title('Press ''s'' to stop. Press any other keys to continue.');
    axis equal;
    axis off;
    hold off;
    
    while (true)
        w = waitforbuttonpress;
        if (~w)
            continue;
        end
        c = get(fig,'CurrentCharacter');
        if (c == 's')
            stop = true;
        end
        break;
    end
    
    close(fig);
end