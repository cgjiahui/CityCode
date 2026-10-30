// Touch laptops and narrow desktop windows keep desktop quality. Device type
// remains stable when a phone rotates. All devices default to full quality;
// the URL records an explicit saver-mode choice.
export function isMobileDevice(device={}) {
  if(device.userAgentData?.mobile)return true;
  const ua=device.userAgent||'';
  return /Android|iPhone|iPad|iPod|Mobile/i.test(ua)
    ||((/Macintosh/i.test(ua)||device.platform==='MacIntel')&&(device.maxTouchPoints||0)>1);
}

export function resolveSceneQuality(search='',device={}) {
  const requested=new URLSearchParams(search).get('quality');
  const preference=['mobile','full'].includes(requested)?requested:'auto';
  const mobileDevice=isMobileDevice(device);
  return {preference,mobileDevice,saver:preference==='mobile'};
}
