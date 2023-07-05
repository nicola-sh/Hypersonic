<?php
require $_SERVER['DOCUMENT_ROOT'] .'/bitrix/modules/main/include/prolog_before.php';

\Bitrix\Main\Loader::includeModule('ipol.aliexpress');

$action  = $_REQUEST['action'];
$code    = $_REQUEST['code'];
$backUrl = '/bitrix/admin/settings.php?lang=ru&mid=ipol.aliexpress&mid_menu=1';
$client  = \Ipol\AliExpress\Api\Client::getInstance();

if ($action == 'logout') {
    \Bitrix\Main\Config\Option::set(IPOLH_ALI_MODULE, 'APP_ACCESS_TOKEN', '{}');
} else {
    if (empty($code)) {
        LocalRedirect($backUrl . '&error=EMPTY_AUTH_CODE');
        exit;
    }

    $data = $client->obtainAccessToken($code);
    $data = json_decode($data, true);

    if (!is_array($data) || empty($data['access_token'])) {       
        LocalRedirect($backUrl .'&error=INVALID_AUTH_CODE');
        exit;
    }

    \Bitrix\Main\Config\Option::set(IPOLH_ALI_MODULE, 'APP_ACCESS_TOKEN', json_encode(array_merge(
        $data, 

        [
            'app_key'    => $client->getAppKey(),
            'secret_key' => $client->getSecretKey(),
        ]
    )));
}

LocalRedirect($backUrl);
