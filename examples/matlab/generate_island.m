function islandMask = generate_island(width, height, threshold)
    % Utility function to generate an island-looking inhomogeneity.
    if nargin < 1 || isempty(width), width = 200; end
    if nargin < 2 || isempty(height), height = 200; end
    if nargin < 3 || isempty(threshold), threshold = 0.3; end
    
    rng('shuffle'); 
    rawNoise = rand(height, width);
    
    % Smooth the noise to create organic shapes
    sigma = 8; 
    smoothNoise = imfilter(rawNoise, fspecial('gaussian', sigma*3, sigma), 'replicate');
    smoothNoise = (smoothNoise - min(smoothNoise(:))) / (max(smoothNoise(:)) - min(smoothNoise(:)));
    
    % Geographic falloff to prevent clipping at the borders
    [X, Y] = meshgrid(1:width, 1:height);
    centerDist = sqrt((X - width/2).^2 + (Y - height/2).^2);
    falloffMap = max(0, 1 - (centerDist / (min(width, height) * 0.6))); 
    
    % Mask and filter
    islandMask = (smoothNoise .* falloffMap) > threshold;
    islandMask = imfill(islandMask, 'holes');
    
    % Allow small islands
    minIslandSize = round(80/(200*200) * width * height);
    islandMask = bwareaopen(islandMask, minIslandSize);
    islandMask = imgaussfilt(double(islandMask), 2.0);
end